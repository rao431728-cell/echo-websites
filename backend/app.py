import os
import re
import json
import time
import anthropic
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max for image uploads
CORS(app)

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def call_claude(system_prompt, user_prompt, max_tokens=4096):
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def parse_json_response(raw, fallback=None):
    if fallback is None:
        fallback = {}

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        fragment = stripped[start : end + 1]
        try:
            return json.loads(fragment)
        except (json.JSONDecodeError, TypeError):
            pass

        cleaned = re.sub(r",\s*([}\]])", r"\1", fragment)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

        cleaned = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group(0).replace("\n", "\\n"), cleaned, flags=re.DOTALL)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

    return fallback


def extract_html(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:html)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    for marker in ("<!DOCTYPE", "<!doctype", "<html"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    return text


def build_enhanced_prompt(data):
    """Build a rich prompt from structured inputs + free-text description."""
    parts = []

    prompt_text = data.get("prompt", "").strip()
    business_name = data.get("business_name", "").strip()
    website_type = data.get("website_type", "").strip()
    style = data.get("style", "").strip()
    color_preference = data.get("color_preference", "").strip()
    features = data.get("features", [])

    if prompt_text:
        parts.append(f"Description: {prompt_text}")

    if business_name:
        parts.append(f"Business/Project name: {business_name}")

    if website_type:
        type_labels = {
            "portfolio": "Portfolio / Showcase",
            "landing-page": "Landing Page",
            "restaurant": "Restaurant / Food & Drink",
            "business": "Business / Corporate",
            "agency": "Creative Agency",
            "saas": "SaaS / Software Product",
            "ecommerce": "E-commerce / Online Store",
            "blog": "Blog / Magazine",
            "event": "Event / Conference",
            "personal": "Personal / Resume",
            "fitness": "Fitness / Wellness",
            "real-estate": "Real Estate",
        }
        parts.append(f"Website type: {type_labels.get(website_type, website_type)}")

    if style:
        style_descriptions = {
            "professional": "Professional — clean, corporate, trustworthy, structured layouts",
            "creative": "Creative — artistic, expressive, unique layouts, bold visual choices",
            "minimal": "Minimal — lots of whitespace, simple, elegant, less is more",
            "bold": "Bold — high contrast, large typography, energetic, attention-grabbing",
            "luxury": "Luxury — premium, elegant, sophisticated, dark tones with gold/rich accents",
            "tech": "Tech — futuristic, neon accents on dark backgrounds, geometric, cutting-edge",
        }
        parts.append(f"Design style: {style_descriptions.get(style, style)}")

    if color_preference and color_preference != "auto":
        palette_descriptions = {
            "midnight": "Dark purples and deep blues — mysterious, elegant night-sky mood",
            "ocean": "Deep blues and teals — calm, professional, trustworthy",
            "ember": "Warm reds, oranges, and dark backgrounds — passionate, energetic, bold",
            "forest": "Greens and earthy tones — natural, growth-oriented, fresh",
            "rose": "Pinks and magentas — creative, feminine, modern, playful",
            "mono": "Black, white, and grays — ultra-clean, editorial, timeless",
        }
        parts.append(f"Color palette: {palette_descriptions.get(color_preference, color_preference)}")

    if features:
        feature_labels = {
            "contact": "Contact form section",
            "gallery": "Image gallery / portfolio grid",
            "testimonials": "Client testimonials / reviews",
            "pricing": "Pricing table with tiers",
            "faq": "FAQ section with accordion",
            "team": "Team members grid",
            "newsletter": "Newsletter signup section",
            "stats": "Statistics / numbers section",
        }
        feature_list = [feature_labels.get(f, f) for f in features]
        parts.append(f"MUST include these sections: {', '.join(feature_list)}")

    return "\n".join(parts)


def agent_intent_analyzer(user_prompt):
    system = (
        "You are a senior creative strategist. Given a user's website request, analyze their needs precisely.\n\n"
        "CRITICAL: Follow the user's request EXACTLY. If they say 'photographer portfolio', the website_type MUST be 'portfolio' "
        "and everything must be tailored to a photographer. If they say 'restaurant', every detail must be restaurant-specific. "
        "Do NOT generalize or deviate from what they asked for.\n\n"
        "Return a JSON object with:\n"
        '- "website_type": string (match what the user asked for exactly)\n'
        '- "audience": string (specific target demographic for THIS type of business)\n'
        '- "goal": string (primary conversion goal appropriate for THIS business)\n'
        '- "tone": string (visual/verbal tone — must match the style they requested)\n'
        '- "key_features": list of 8-12 strings (features SPECIFIC to this exact business type. '
        "A photographer needs: hero with portfolio image, gallery grid, lightbox, about/bio, client list, testimonials, booking CTA. "
        "A restaurant needs: hero with food, menu with categories/prices, reservation, chef story, hours/location. "
        "A SaaS needs: product hero, feature grid, pricing tiers, integrations, testimonials, FAQ. "
        "Always include: sticky nav, mobile menu, footer, contact section)\n"
        '- "industry": string\n'
        '- "design_inspiration": string (describe the exact visual style — specific to their industry and preferences)\n'
        '- "emotional_response": string (what a visitor should feel)\n\n'
        "If the user specified a business name, use it throughout.\n"
        "If the user specified a style (professional, creative, minimal, bold, luxury, tech), match it exactly.\n"
        "If the user specified a color palette preference, acknowledge it.\n\n"
        "Return ONLY valid JSON, no markdown fences."
    )
    raw = call_claude(system, user_prompt, max_tokens=1024)
    fallback = {
        "website_type": "website",
        "audience": "general",
        "goal": "inform visitors",
        "tone": "professional",
        "key_features": ["hero section", "about section", "services section", "testimonials", "contact section", "footer"],
        "industry": "general",
        "design_inspiration": "clean modern website",
        "emotional_response": "trust and interest",
    }
    return parse_json_response(raw, fallback)


def agent_site_architect(intent):
    system = (
        "You are a senior UX architect. Design the page structure for this SPECIFIC website type.\n\n"
        "CRITICAL: The sections MUST be tailored to the exact website type and industry. "
        "A photographer portfolio has DIFFERENT sections than a restaurant or a SaaS product. "
        "Do NOT use generic sections. Every section must serve this specific business.\n\n"
        "Return a JSON object with:\n"
        '- "title": string (use the business name if provided, otherwise create a memorable one)\n'
        '- "nav_items": list of 4-6 strings (navigation labels specific to this business)\n'
        '- "sections": list of 7-10 objects, each with:\n'
        '  - "id": string (html id, lowercase hyphenated)\n'
        '  - "type": string (hero, about, gallery, services, testimonials, contact, footer, features, pricing, cta, stats, faq, team, portfolio, process, clients, menu, case-studies)\n'
        '  - "purpose": string (what this section does for the conversion funnel)\n'
        '  - "layout": string (exact layout description: "2-column with image left", "3-card grid", "masonry grid", "alternating zigzag")\n'
        '  - "content_hints": string (specific content: how many items, what kind)\n'
        '  - "visual_effect": string (parallax, fade-in-up, staggered-reveal, counter-animation, gradient-text)\n\n'
        "RULES:\n"
        "- If the user requested specific sections (gallery, pricing, FAQ, etc.), INCLUDE them\n"
        "- Hero is always first, footer is always last\n"
        "- Section types must match the business: a restaurant gets 'menu', a portfolio gets 'gallery', SaaS gets 'pricing'\n"
        "- Use the actual business name in the title field\n"
        "- Nav items should be SHORT (1-2 words) and specific to this site\n\n"
        "Return ONLY valid JSON, no markdown fences."
    )
    raw = call_claude(system, json.dumps(intent), max_tokens=2048)
    fallback = {
        "title": intent.get("website_type", "My Website").title(),
        "nav_items": ["Home", "About", "Services", "Contact"],
        "sections": [
            {"id": "hero", "type": "hero", "purpose": "First impression", "layout": "centered", "content_hints": "Main headline and CTA", "visual_effect": "fade-in"},
            {"id": "about", "type": "about", "purpose": "Tell the story", "layout": "2-column", "content_hints": "Bio and image", "visual_effect": "fade-in-up"},
            {"id": "services", "type": "services", "purpose": "Show offerings", "layout": "3-card grid", "content_hints": "3-4 service cards", "visual_effect": "staggered reveal"},
            {"id": "testimonials", "type": "testimonials", "purpose": "Social proof", "layout": "carousel", "content_hints": "3 quotes", "visual_effect": "fade-in"},
            {"id": "contact", "type": "contact", "purpose": "Conversion", "layout": "centered", "content_hints": "Contact form or CTA", "visual_effect": "fade-in-up"},
            {"id": "footer", "type": "footer", "purpose": "Navigation and info", "layout": "3-column", "content_hints": "Links and social", "visual_effect": "none"},
        ],
    }
    return parse_json_response(raw, fallback)


def agent_design(intent, architecture):
    system = (
        "You are an award-winning visual designer. Create a design system for this SPECIFIC website.\n\n"
        "CRITICAL: The design must match the website type, industry, and any style preferences the user specified. "
        "A photographer portfolio looks NOTHING like a SaaS landing page or a restaurant site. "
        "Match the mood exactly.\n\n"
        "Return a JSON object with:\n\n"
        "COLOR PALETTE (all hex):\n"
        '- "primary_color": main brand color (tailored to this specific industry and mood)\n'
        '- "primary_light": lighter tint for hovers\n'
        '- "primary_dark": darker shade for active states\n'
        '- "secondary_color": complementary color\n'
        '- "accent_color": high-contrast pop color for CTAs\n'
        '- "background_color": page background (dark for moody/tech/luxury, light for minimal/clean)\n'
        '- "background_alt": alternate section background\n'
        '- "surface_color": card backgrounds\n'
        '- "text_color": primary text (must have 4.5:1+ contrast against background)\n'
        '- "text_secondary": secondary text\n'
        '- "text_on_primary": text on primary_color backgrounds\n'
        '- "gradient_start": hero/CTA gradient start\n'
        '- "gradient_end": hero/CTA gradient end\n'
        '- "gradient_angle": string (e.g. "135deg")\n\n'
        "TYPOGRAPHY:\n"
        '- "heading_font": Google Fonts name (match the mood — "Playfair Display" for luxury, "Space Grotesk" for tech, "Sora" for modern)\n'
        '- "body_font": Google Fonts name (legible — "Inter", "DM Sans", "Plus Jakarta Sans")\n'
        '- "hero_size": string (fluid clamp, e.g. "clamp(2.5rem, 6vw, 5rem)")\n'
        '- "h2_size": string\n'
        '- "body_size": string\n'
        '- "heading_weight": string\n'
        '- "heading_letter_spacing": string\n\n'
        "SPACING & SHAPE:\n"
        '- "border_radius": string\n'
        '- "section_padding": string\n'
        '- "card_padding": string\n'
        '- "max_width": string\n\n'
        "EFFECTS:\n"
        '- "shadow_sm": CSS box-shadow\n'
        '- "shadow_lg": CSS box-shadow\n'
        '- "glass_effect": boolean\n'
        '- "use_gradients": boolean\n'
        '- "style_notes": string (2-3 sentences describing the visual direction)\n\n'
        "Return ONLY valid JSON, no markdown fences."
    )
    prompt = json.dumps({"intent": intent, "architecture": architecture})
    raw = call_claude(system, prompt, max_tokens=2048)
    fallback = {
        "primary_color": "#6c5ce7", "primary_light": "#a29bfe", "primary_dark": "#4a3db8",
        "secondary_color": "#00cec9", "accent_color": "#fd79a8",
        "background_color": "#0a0a0f", "background_alt": "#12121a",
        "surface_color": "#1a1a26", "text_color": "#e4e4ef", "text_secondary": "#7a7a8e",
        "text_on_primary": "#ffffff",
        "gradient_start": "#6c5ce7", "gradient_end": "#a29bfe", "gradient_angle": "135deg",
        "heading_font": "Inter", "body_font": "Inter",
        "hero_size": "clamp(2.5rem, 6vw, 5rem)", "h2_size": "clamp(1.8rem, 4vw, 3rem)",
        "body_size": "1.05rem", "heading_weight": "700", "heading_letter_spacing": "-0.02em",
        "border_radius": "12px", "section_padding": "100px", "card_padding": "32px", "max_width": "1200px",
        "shadow_sm": "0 2px 8px rgba(0,0,0,0.1)", "shadow_lg": "0 8px 32px rgba(0,0,0,0.2)",
        "glass_effect": False, "use_gradients": True,
        "style_notes": "Modern dark theme with purple accents",
    }
    return parse_json_response(raw, fallback)


def agent_copywriter(intent, architecture):
    system = (
        "You are a world-class conversion copywriter. Write copy for this SPECIFIC business.\n\n"
        "CRITICAL: Every piece of copy must be tailored to this exact business type and industry. "
        "If it's a photographer, write about photography. If it's a restaurant, write about food and dining. "
        "If a business name was provided, USE IT in headlines and CTAs. "
        "NEVER write generic copy. Every sentence must be specific to this business.\n\n"
        "Return a JSON object where each key is a section id from the architecture, "
        "and the value is an object with:\n"
        '- "headline": string (specific to this business — max 8 words, powerful)\n'
        '- "subheadline": string (1-2 sentences expanding on the headline)\n'
        '- "body": string (1-2 paragraphs, specific and vivid)\n'
        '- "cta_text": string (action-oriented, specific to this business: "Book a Session", "Reserve a Table", "Start Free Trial")\n'
        '- "cta_secondary": string (optional secondary CTA)\n'
        '- "items": list of objects with "title", "description", "icon_name" (Font Awesome 6), "stat" (optional)\n'
        '- "testimonials": list of 3 objects with "quote", "author", "role", "company"\n\n'
        "RULES:\n"
        "- Use the business name in the hero headline\n"
        "- Headlines create curiosity or state a bold benefit specific to the business\n"
        "- CTAs must be specific: 'View Gallery' not 'Learn More', 'Reserve Your Table' not 'Click Here'\n"
        "- Testimonials should reference specific results relevant to this business\n"
        "- Items/services must be real offerings this type of business would have\n"
        "- Stats should be believable numbers for this industry\n"
        "- NEVER use lorem ipsum or generic filler\n\n"
        "Return ONLY valid JSON, no markdown fences."
    )
    prompt = json.dumps({"intent": intent, "architecture": architecture})
    raw = call_claude(system, prompt, max_tokens=8192)
    sections = architecture.get("sections", [])
    fallback = {}
    for section in sections:
        sid = section.get("id", "section")
        fallback[sid] = {
            "headline": section.get("type", "Section").replace("-", " ").title(),
            "subheadline": section.get("purpose", ""),
            "body": section.get("content_hints", ""),
            "cta_text": "Get Started",
        }
    return parse_json_response(raw, fallback)


def agent_component_builder(intent, architecture, design, copy):
    """Hardcoded skeleton with enhanced 3D/interactive effects + model-generated sections."""

    sections = architecture.get("sections", [])
    title = architecture.get("title", "Website")
    heading_font = design.get("heading_font", "Inter")
    body_font = design.get("body_font", "Inter")
    nav_items = architecture.get("nav_items", [s.get("id", "").replace("-", " ").title() for s in sections[:5]])

    fonts = set([heading_font, body_font])
    font_url = "https://fonts.googleapis.com/css2?" + "&".join(
        f"family={f.replace(' ', '+')}:wght@300;400;500;600;700;800;900" for f in fonts
    ) + "&display=swap"

    d = design
    primary = d.get("primary_color", "#6c5ce7")
    primary_light = d.get("primary_light", primary)
    primary_dark = d.get("primary_dark", "#4a3db8")
    secondary = d.get("secondary_color", "#00cec9")
    accent = d.get("accent_color", "#fd79a8")
    bg = d.get("background_color", "#0a0a0f")
    bg_alt = d.get("background_alt", "#12121a")
    surface = d.get("surface_color", "#1a1a26")
    text_color = d.get("text_color", "#e4e4ef")
    text_sec = d.get("text_secondary", "#7a7a8e")
    text_on_primary = d.get("text_on_primary", "#ffffff")
    grad_start = d.get("gradient_start", primary)
    grad_end = d.get("gradient_end", primary_light)
    grad_angle = d.get("gradient_angle", "135deg")
    radius = d.get("border_radius", "12px")
    shadow_sm = d.get("shadow_sm", "0 2px 8px rgba(0,0,0,0.1)")
    shadow_lg = d.get("shadow_lg", "0 8px 32px rgba(0,0,0,0.2)")

    nav_html = ""
    for i, item in enumerate(nav_items):
        sid = sections[i].get("id", "") if i < len(sections) else ""
        nav_html += f'<a href="#{sid}">{item}</a>\n'

    skeleton_top = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="{font_url}" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
:root {{
  --primary: {primary};
  --primary-light: {primary_light};
  --primary-dark: {primary_dark};
  --secondary: {secondary};
  --accent: {accent};
  --bg: {bg};
  --bg-alt: {bg_alt};
  --surface: {surface};
  --text: {text_color};
  --text-sec: {text_sec};
  --text-on-primary: {text_on_primary};
  --grad-start: {grad_start};
  --grad-end: {grad_end};
  --grad-angle: {grad_angle};
  --radius: {radius};
  --shadow-sm: {shadow_sm};
  --shadow-lg: {shadow_lg};
  --heading-font: '{heading_font}', sans-serif;
  --body-font: '{body_font}', sans-serif;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ font-family: var(--body-font); color: var(--text); background: var(--bg); line-height:1.6; overflow-x:hidden; }}
.container {{ max-width:1200px; margin:0 auto; padding:0 24px; position:relative; }}

/* === PAGE LOADER === */
.page-loader {{ position:fixed; inset:0; background:var(--bg); display:flex; align-items:center; justify-content:center;
  z-index:99999; transition: opacity 0.6s, visibility 0.6s; }}
.page-loader.hidden {{ opacity:0; visibility:hidden; pointer-events:none; }}
.loader-ring {{ width:40px; height:40px; border:3px solid rgba(255,255,255,0.1); border-top-color:var(--primary);
  border-radius:50%; animation:loader-spin 0.8s linear infinite; }}
@keyframes loader-spin {{ to {{ transform:rotate(360deg); }} }}

/* === SCROLL PROGRESS BAR === */
.scroll-progress {{ position:fixed; top:0; left:0; height:3px; background:linear-gradient(90deg, var(--grad-start), var(--grad-end));
  z-index:10001; width:0; transition:width 0.1s linear; }}

/* === CUSTOM CURSOR === */
.cursor-glow {{ position:fixed; width:24px; height:24px; border-radius:50%; pointer-events:none; z-index:9999;
  background:radial-gradient(circle, var(--primary), transparent 70%); opacity:0.4; transform:translate(-50%,-50%);
  transition:width 0.3s, height 0.3s, opacity 0.3s; mix-blend-mode:screen; }}
.cursor-glow.hover {{ width:48px; height:48px; opacity:0.6; }}
@media(max-width:768px) {{ .cursor-glow {{ display:none; }} }}

/* === BACKGROUND EFFECTS === */
/* Noise grain overlay */
body::before {{ content:''; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:0.03;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }}

/* Animated aurora gradient blobs */
.aurora {{ position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }}
.aurora-blob {{ position:absolute; border-radius:50%; filter:blur(80px); opacity:0.12;
  animation:aurora-drift 12s ease-in-out infinite alternate; }}
.aurora-blob:nth-child(1) {{ width:600px; height:600px; top:-200px; left:-100px;
  background:var(--primary); animation-duration:14s; }}
.aurora-blob:nth-child(2) {{ width:500px; height:500px; bottom:-150px; right:-100px;
  background:var(--grad-end); animation-delay:3s; animation-duration:16s; }}
.aurora-blob:nth-child(3) {{ width:400px; height:400px; top:40%; left:50%;
  background:var(--secondary); animation-delay:6s; animation-duration:18s; }}
@keyframes aurora-drift {{
  0% {{ transform:translate(0,0) scale(1); }}
  33% {{ transform:translate(40px,-30px) scale(1.1); }}
  66% {{ transform:translate(-20px,40px) scale(0.9); }}
  100% {{ transform:translate(30px,20px) scale(1.05); }}
}}

/* Particle canvas */
.particle-canvas {{ position:fixed; inset:0; z-index:0; pointer-events:none; }}

/* Dot grid background on sections */
.section-dark::before {{ content:''; position:absolute; inset:0; pointer-events:none;
  background-image:radial-gradient(rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size:32px 32px; z-index:0; }}
.section-light::before {{ content:''; position:absolute; inset:0; pointer-events:none;
  background-image:radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
  background-size:32px 32px; z-index:0; }}

/* Radial glow accents on sections */
.section-dark::after {{ content:''; position:absolute; top:0; left:50%; transform:translateX(-50%);
  width:800px; height:400px; pointer-events:none;
  background:radial-gradient(ellipse, var(--primary), transparent 70%); opacity:0.04; z-index:0; }}
.section-light::after {{ content:''; position:absolute; bottom:0; right:10%;
  width:600px; height:300px; pointer-events:none;
  background:radial-gradient(ellipse, var(--grad-end), transparent 70%); opacity:0.05; z-index:0; }}

/* Mouse-follow glow on sections */
.section-dark, .section-light {{ --glow-x:50%; --glow-y:50%; }}
.section-dark > .container::before, .section-light > .container::before {{ content:''; position:absolute;
  width:500px; height:500px; border-radius:50%; pointer-events:none; z-index:0;
  left:var(--glow-x); top:var(--glow-y); transform:translate(-50%,-50%);
  background:radial-gradient(circle, var(--primary), transparent 70%); opacity:0.03;
  transition:left 0.3s ease, top 0.3s ease; }}

/* Ensure content sits above backgrounds */
.section-dark > *, .section-light > * {{ position:relative; z-index:1; }}

/* === SECTION THEMES === */
.section-dark {{ background: var(--bg); color: #ffffff; padding: 90px 0; position:relative; overflow:hidden; }}
.section-dark h1, .section-dark h2, .section-dark h3 {{ color: #ffffff; font-family: var(--heading-font); }}
.section-dark p, .section-dark li, .section-dark span {{ color: rgba(255,255,255,0.85); }}
.section-dark .card {{ background: var(--surface); color: #ffffff; border: 1px solid rgba(255,255,255,0.08);
  position:relative; overflow:hidden; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }}
.section-dark .card h3 {{ color: #ffffff; }}
.section-dark .card p {{ color: rgba(255,255,255,0.8); }}

.section-light {{ background: var(--bg-alt); color: #ffffff; padding: 90px 0; position:relative; overflow:hidden; }}
.section-light h1, .section-light h2, .section-light h3 {{ color: #ffffff; font-family: var(--heading-font); }}
.section-light p, .section-light li, .section-light span {{ color: rgba(255,255,255,0.8); }}
.section-light .card {{ background: rgba(255,255,255,0.03); color: #ffffff;
  border: 1px solid rgba(255,255,255,0.08); position:relative; overflow:hidden;
  backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }}
.section-light .card h3 {{ color: #ffffff; }}
.section-light .card p {{ color: rgba(255,255,255,0.75); }}

/* === HERO === */
.hero {{ min-height:100vh; display:flex; align-items:center; position:relative; overflow:hidden;
  background: linear-gradient(var(--grad-angle), var(--grad-start), var(--grad-end)); }}
.hero::before {{ content:''; position:absolute; inset:0;
  background: radial-gradient(ellipse at 20% 50%, rgba(255,255,255,0.15) 0%, transparent 50%),
              radial-gradient(ellipse at 80% 20%, rgba(255,255,255,0.08) 0%, transparent 40%),
              radial-gradient(ellipse at 50% 80%, var(--secondary), transparent 50%);
  opacity:0.6; animation:hero-bg-shift 10s ease-in-out infinite alternate; }}
@keyframes hero-bg-shift {{
  0% {{ transform:scale(1) translate(0,0); opacity:0.6; }}
  100% {{ transform:scale(1.1) translate(2%,-2%); opacity:0.4; }}
}}
.hero::after {{ content:''; position:absolute; inset:0; background:rgba(0,0,0,0.15);
  background-image:radial-gradient(rgba(255,255,255,0.02) 1px, transparent 1px);
  background-size:24px 24px; }}
.hero .container {{ position:relative; z-index:2; }}
.hero h1 {{ font-size: clamp(2.8rem, 7vw, 5.5rem); font-weight:900; line-height:1.05; margin-bottom:24px;
  color:#fff; letter-spacing:-0.03em; }}
.hero p {{ font-size: clamp(1.05rem, 2vw, 1.3rem); margin-bottom:36px; color:rgba(255,255,255,0.85);
  max-width:600px; line-height:1.7; }}
.hero .btn {{ margin-right:12px; margin-bottom:12px; }}

/* Hero floating shapes */
.hero-shapes {{ position:absolute; inset:0; overflow:hidden; pointer-events:none; z-index:1; }}
.hero-shape {{ position:absolute; border-radius:50%; opacity:0.08; background:rgba(255,255,255,0.5);
  animation:float-shape 8s ease-in-out infinite; }}
.hero-shape:nth-child(1) {{ width:300px; height:300px; top:-50px; right:-50px; animation-delay:0s; }}
.hero-shape:nth-child(2) {{ width:200px; height:200px; bottom:10%; left:5%; animation-delay:2s; }}
.hero-shape:nth-child(3) {{ width:150px; height:150px; top:20%; right:20%; animation-delay:4s; }}
.hero-shape:nth-child(4) {{ width:100px; height:100px; bottom:30%; right:10%; animation-delay:1s; }}
@keyframes float-shape {{
  0%,100% {{ transform:translateY(0) rotate(0deg) scale(1); }}
  33% {{ transform:translateY(-20px) rotate(5deg) scale(1.05); }}
  66% {{ transform:translateY(10px) rotate(-3deg) scale(0.95); }}
}}

/* === TYPOGRAPHY === */
h2 {{ font-size: clamp(2rem, 4.5vw, 3.2rem); font-weight:800; margin-bottom:16px;
  font-family: var(--heading-font); letter-spacing:-0.02em; line-height:1.15; }}
h3 {{ font-size: 1.3rem; font-weight:600; margin-bottom:10px; font-family: var(--heading-font); }}
.section-header {{ text-align:center; max-width:700px; margin:0 auto 48px; }}
.section-header p {{ font-size:1.1rem; line-height:1.7; }}
.gradient-text {{ background:linear-gradient(var(--grad-angle), var(--grad-start), var(--grad-end));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }}

/* === GRID & CARDS === */
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); gap:32px; }}
.card {{ padding:36px; border-radius: var(--radius); transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
  position:relative; overflow:hidden; }}
.card::before {{ content:''; position:absolute; inset:0; border-radius:inherit; opacity:0;
  background:linear-gradient(var(--grad-angle), var(--primary), transparent); transition:opacity 0.4s; z-index:0; }}
.card:hover::before {{ opacity:0.05; }}
.card:hover {{ transform:translateY(-12px) scale(1.02); box-shadow: var(--shadow-lg), 0 0 40px rgba(0,0,0,0.1); }}
.card > * {{ position:relative; z-index:1; }}
.card i {{ font-size:2.2rem; color: var(--primary); margin-bottom:18px; display:block;
  background:linear-gradient(var(--grad-angle), var(--grad-start), var(--grad-end));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.card .card-icon {{ width:56px; height:56px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(var(--grad-angle), rgba(var(--primary), 0.1), rgba(var(--primary), 0.05));
  margin-bottom:20px; }}

/* === BUTTONS === */
.btn {{ display:inline-flex; align-items:center; gap:8px; padding:16px 36px; border-radius:8px; font-weight:600;
  text-decoration:none; transition:all 0.3s cubic-bezier(0.4,0,0.2,1); cursor:pointer; border:none;
  font-size:1rem; font-family:var(--body-font); position:relative; overflow:hidden; }}
.btn-primary {{ background:linear-gradient(135deg, var(--primary), var(--primary-dark)); color:var(--text-on-primary);
  box-shadow:0 4px 20px rgba(0,0,0,0.2); }}
.btn-primary:hover {{ transform:translateY(-3px) scale(1.03); box-shadow:0 8px 30px rgba(0,0,0,0.3);
  filter:brightness(1.1); }}
.btn-primary::after {{ content:''; position:absolute; inset:-50%; background:linear-gradient(90deg,
  transparent, rgba(255,255,255,0.1), transparent); transform:rotate(45deg) translateX(-100%);
  transition:transform 0.6s; }}
.btn-primary:hover::after {{ transform:rotate(45deg) translateX(100%); }}
.btn-outline {{ background:transparent; border:2px solid rgba(255,255,255,0.25); color:#ffffff; }}
.btn-outline:hover {{ background:rgba(255,255,255,0.08); border-color:rgba(255,255,255,0.4);
  transform:translateY(-3px); }}

/* === IMAGES === */
img {{ max-width:100%; height:auto; border-radius: var(--radius); display:block; }}
.img-reveal {{ clip-path:inset(0 100% 0 0); transition:clip-path 1s cubic-bezier(0.77,0,0.175,1); }}
.img-reveal.visible {{ clip-path:inset(0 0 0 0); }}
.img-cover {{ width:100%; height:100%; object-fit:cover; }}
.img-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(250px,1fr)); gap:16px; }}
.img-grid img {{ width:100%; height:250px; object-fit:cover; border-radius:var(--radius);
  transition:transform 0.4s, filter 0.4s; }}
.img-grid img:hover {{ transform:scale(1.03); filter:brightness(1.1); }}

/* === TWO COLUMN LAYOUTS === */
.two-col {{ display:flex; gap:48px; align-items:center; }}
.two-col > * {{ flex:1; }}
@media(max-width:768px) {{ .two-col {{ flex-direction:column; gap:32px; }} }}

/* === BADGES & TAGS === */
.badge {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600;
  background:var(--accent-surface, rgba(124,92,252,0.1)); color:var(--primary); border:1px solid rgba(124,92,252,0.2);
  text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px; }}
.tag {{ display:inline-block; padding:3px 10px; border-radius:6px; font-size:0.7rem;
  background:rgba(255,255,255,0.05); color:rgba(255,255,255,0.6); margin:2px; }}

/* === AVATAR === */
.avatar {{ width:48px; height:48px; border-radius:50%; object-fit:cover; border:2px solid var(--primary); }}
.avatar-sm {{ width:36px; height:36px; }}

/* === TESTIMONIAL CARD EXTRAS === */
.testimonial-author {{ display:flex; align-items:center; gap:12px; margin-top:16px; }}

/* === IMAGE OVERLAY === */
.img-overlay-wrap {{ position:relative; overflow:hidden; border-radius:var(--radius); }}
.img-overlay-wrap img {{ width:100%; height:300px; object-fit:cover; transition:transform 0.5s; }}
.img-overlay-wrap:hover img {{ transform:scale(1.08); }}
.img-overlay {{ position:absolute; inset:0; background:linear-gradient(to top, rgba(0,0,0,0.7), transparent);
  display:flex; align-items:flex-end; padding:20px; opacity:0; transition:opacity 0.4s; }}
.img-overlay-wrap:hover .img-overlay {{ opacity:1; }}
.img-overlay p {{ color:#fff; font-weight:600; font-size:1rem; }}

/* === ANIMATIONS === */
.animate {{ opacity:0; transform:translateY(40px); transition: opacity 0.7s cubic-bezier(0.4,0,0.2,1),
  transform 0.7s cubic-bezier(0.4,0,0.2,1); }}
.animate.visible {{ opacity:1; transform:translateY(0); }}
.animate-left {{ opacity:0; transform:translateX(-40px); transition: opacity 0.7s ease, transform 0.7s ease; }}
.animate-left.visible {{ opacity:1; transform:translateX(0); }}
.animate-right {{ opacity:0; transform:translateX(40px); transition: opacity 0.7s ease, transform 0.7s ease; }}
.animate-right.visible {{ opacity:1; transform:translateX(0); }}
.animate-scale {{ opacity:0; transform:scale(0.9); transition: opacity 0.7s ease, transform 0.7s ease; }}
.animate-scale.visible {{ opacity:1; transform:scale(1); }}

/* === NAV === */
header {{ position:fixed; top:0; left:0; right:0; z-index:1000; padding:18px 0;
  background:transparent; transition: all 0.4s cubic-bezier(0.4,0,0.2,1); }}
header.scrolled {{ background:rgba({int(bg[1:3],16)},{int(bg[3:5],16)},{int(bg[5:7],16)},0.92);
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  box-shadow:0 2px 30px rgba(0,0,0,0.3); padding:12px 0; }}
nav {{ display:flex; align-items:center; justify-content:space-between; }}
.logo {{ font-family: var(--heading-font); font-size:1.5rem; font-weight:800; color:#fff;
  text-decoration:none; letter-spacing:-0.02em; }}
.nav-links {{ display:flex; gap:28px; list-style:none; }}
.nav-links a {{ color: rgba(255,255,255,0.75); text-decoration:none; font-weight:500; font-size:0.95rem;
  transition: color 0.3s, transform 0.3s; position:relative; }}
.nav-links a:hover {{ color:#fff; transform:translateY(-1px); }}
.nav-links a::after {{ content:''; position:absolute; bottom:-4px; left:0; width:0; height:2px;
  background:var(--primary); transition:width 0.3s; border-radius:2px; }}
.nav-links a:hover::after {{ width:100%; }}
.hamburger {{ display:none; flex-direction:column; gap:5px; cursor:pointer; background:none; border:none; padding:8px; }}
.hamburger span {{ width:24px; height:2px; background:#fff; transition:all 0.3s; border-radius:2px; }}

/* === FOOTER === */
footer {{ background: var(--bg); color:rgba(255,255,255,0.7); padding:100px 0 40px; }}
footer h3 {{ color:#ffffff; margin-bottom:18px; font-size:1.1rem; }}
footer a {{ color:rgba(255,255,255,0.5); text-decoration:none; transition:color 0.3s; }}
footer a:hover {{ color: var(--primary-light); }}
.footer-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:48px; }}
.footer-bottom {{ border-top:1px solid rgba(255,255,255,0.08); margin-top:48px; padding-top:28px;
  display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; }}
.social-links {{ display:flex; gap:18px; }}
.social-links a {{ font-size:1.3rem; color:rgba(255,255,255,0.5); transition:all 0.3s; }}
.social-links a:hover {{ color: var(--primary-light); transform:translateY(-3px); }}

/* === TESTIMONIALS === */
blockquote {{ font-style:italic; font-size:1.1rem; line-height:1.8; margin-bottom:18px; position:relative; padding-left:20px;
  border-left:3px solid var(--primary); }}
cite {{ font-style:normal; font-weight:600; font-size:0.95rem; }}

/* === STATS === */
.stat-number {{ font-size: clamp(2.5rem, 5vw, 4rem); font-weight:900; color: var(--primary);
  font-family:var(--heading-font); line-height:1; }}

/* === RESPONSIVE === */
@media(max-width:768px) {{
  .grid {{ grid-template-columns:1fr; }}
  section, .section-dark, .section-light {{ padding:60px 0; }}
  .nav-links {{ display:none; position:fixed; top:0; left:0; right:0; bottom:0;
    background:rgba({int(bg[1:3],16)},{int(bg[3:5],16)},{int(bg[5:7],16)},0.98);
    flex-direction:column; align-items:center; justify-content:center; gap:36px; z-index:999;
    backdrop-filter:blur(20px); }}
  .nav-links.active {{ display:flex; }}
  .nav-links a {{ font-size:1.3rem; }}
  .hamburger {{ display:flex; z-index:1000; }}
  .footer-grid {{ grid-template-columns:1fr; }}
  .footer-bottom {{ flex-direction:column; text-align:center; }}
  .hero h1 {{ font-size:clamp(2rem, 8vw, 3.5rem); }}
}}
@media(prefers-reduced-motion:reduce) {{
  .animate,.animate-left,.animate-right,.animate-scale {{ opacity:1; transform:none; transition:none; }}
  .hero-shape {{ animation:none; }}
  .cursor-glow {{ display:none; }}
}}
</style>
</head>
<body>
<!-- Aurora Background -->
<div class="aurora">
  <div class="aurora-blob"></div>
  <div class="aurora-blob"></div>
  <div class="aurora-blob"></div>
</div>

<!-- Particle Canvas -->
<canvas class="particle-canvas" id="particleCanvas"></canvas>

<!-- Page Loader -->
<div class="page-loader" id="pageLoader">
  <div class="loader-ring"></div>
</div>

<!-- Scroll Progress -->
<div class="scroll-progress" id="scrollProgress"></div>

<!-- Custom Cursor -->
<div class="cursor-glow" id="cursorGlow"></div>

<header>
<div class="container">
<nav>
<a href="#" class="logo">{title}</a>
<div class="nav-links">
{nav_html}
</div>
<button class="hamburger" aria-label="Menu">
<span></span><span></span><span></span>
</button>
</nav>
</div>
</header>

<!-- Hero floating shapes -->
<div class="hero-shapes">
<div class="hero-shape"></div>
<div class="hero-shape"></div>
<div class="hero-shape"></div>
<div class="hero-shape"></div>
</div>

<main>
'''

    skeleton_bottom = '''</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>
<script>
gsap.registerPlugin(ScrollTrigger);

// === PARTICLE SYSTEM ===
(function() {
  const canvas = document.getElementById("particleCanvas");
  if(!canvas) return;
  const ctx = canvas.getContext("2d");
  let particles = [];
  const PARTICLE_COUNT = 60;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  function createParticle() {
    return {
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      size: Math.random() * 2 + 0.5,
      opacity: Math.random() * 0.4 + 0.1
    };
  }
  for(let i = 0; i < PARTICLE_COUNT; i++) particles.push(createParticle());

  function drawParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const style = getComputedStyle(document.documentElement);
    const color = style.getPropertyValue("--primary").trim() || "#6c5ce7";

    particles.forEach((p, i) => {
      p.x += p.vx;
      p.y += p.vy;
      if(p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if(p.y < 0 || p.y > canvas.height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = p.opacity;
      ctx.fill();

      // Connect nearby particles with lines
      for(let j = i + 1; j < particles.length; j++) {
        const dx = p.x - particles[j].x;
        const dy = p.y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if(dist < 120) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = color;
          ctx.globalAlpha = (1 - dist/120) * 0.08;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    });
    ctx.globalAlpha = 1;
    requestAnimationFrame(drawParticles);
  }
  drawParticles();
})();

// === PAGE LOADER ===
window.addEventListener("load", () => {
  setTimeout(() => {
    document.getElementById("pageLoader").classList.add("hidden");
    // Animate hero content in
    gsap.from(".hero h1", {y:60, opacity:0, duration:1, delay:0.2, ease:"power3.out"});
    gsap.from(".hero p", {y:40, opacity:0, duration:0.8, delay:0.5, ease:"power3.out"});
    gsap.from(".hero .btn", {y:30, opacity:0, duration:0.6, delay:0.8, stagger:0.15, ease:"power3.out"});
  }, 300);
});

// === SCROLL PROGRESS ===
window.addEventListener("scroll", () => {
  const scrolled = window.scrollY / (document.documentElement.scrollHeight - window.innerHeight) * 100;
  document.getElementById("scrollProgress").style.width = Math.min(scrolled, 100) + "%";
});

// === CUSTOM CURSOR ===
const cursorGlow = document.getElementById("cursorGlow");
let mouseX = 0, mouseY = 0, cursorX = 0, cursorY = 0;
document.addEventListener("mousemove", e => { mouseX = e.clientX; mouseY = e.clientY; });
function animateCursor() {
  cursorX += (mouseX - cursorX) * 0.15;
  cursorY += (mouseY - cursorY) * 0.15;
  cursorGlow.style.left = cursorX + "px";
  cursorGlow.style.top = cursorY + "px";
  requestAnimationFrame(animateCursor);
}
animateCursor();
document.querySelectorAll("a, .btn, .card").forEach(el => {
  el.addEventListener("mouseenter", () => cursorGlow.classList.add("hover"));
  el.addEventListener("mouseleave", () => cursorGlow.classList.remove("hover"));
});

// === SCROLL REVEAL (IntersectionObserver) ===
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if(e.isIntersecting) { e.target.classList.add("visible"); observer.unobserve(e.target); }
  });
}, {threshold:0.1, rootMargin:"0px 0px -60px 0px"});
document.querySelectorAll(".animate, .animate-left, .animate-right, .animate-scale, .img-reveal").forEach(el => observer.observe(el));

// === GSAP HEADING ANIMATIONS ===
gsap.utils.toArray("section h2, .section-dark h2, .section-light h2").forEach(h => {
  gsap.from(h, {y:60, opacity:0, duration:0.9, ease:"power3.out",
    scrollTrigger:{trigger:h, start:"top 85%"}});
});

// === GSAP CARD STAGGER ===
gsap.utils.toArray(".grid").forEach(grid => {
  const cards = grid.querySelectorAll(".card");
  if(cards.length) {
    gsap.from(cards, {
      y:80, opacity:0, duration:0.7, stagger:0.12, ease:"power3.out",
      scrollTrigger:{trigger:grid, start:"top 82%"}
    });
  }
});

// === HERO PARALLAX ===
const heroEl = document.querySelector(".hero");
if(heroEl) {
  gsap.to(".hero", {yPercent:-20, ease:"none", scrollTrigger:{trigger:".hero", scrub:1.5}});
  gsap.to(".hero-shapes", {yPercent:-30, ease:"none", scrollTrigger:{trigger:".hero", scrub:1}});
}

// === STICKY NAV ===
window.addEventListener("scroll", () => {
  const h = document.querySelector("header");
  if(h) h.classList.toggle("scrolled", window.scrollY > 80);
});

// === MOBILE MENU ===
const hamburger = document.querySelector(".hamburger");
const navLinks = document.querySelector(".nav-links");
if(hamburger && navLinks) {
  hamburger.addEventListener("click", () => {
    navLinks.classList.toggle("active");
    hamburger.classList.toggle("active");
  });
}

// === SMOOTH SCROLL ===
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    e.preventDefault();
    const t = document.querySelector(a.getAttribute("href"));
    if(t) { t.scrollIntoView({behavior:"smooth"}); }
    if(navLinks) navLinks.classList.remove("active");
  });
});

// === 3D CARD TILT ===
document.querySelectorAll(".card").forEach(card => {
  card.addEventListener("mousemove", e => {
    const r = card.getBoundingClientRect();
    const x = (e.clientX - r.left)/r.width - 0.5;
    const y = (e.clientY - r.top)/r.height - 0.5;
    card.style.transform = `perspective(1000px) rotateY(${x*12}deg) rotateX(${y*-12}deg) translateY(-12px) scale(1.02)`;
    card.style.transition = "transform 0.1s ease";
  });
  card.addEventListener("mouseleave", () => {
    card.style.transform = "";
    card.style.transition = "transform 0.5s cubic-bezier(0.4,0,0.2,1)";
  });
});

// === MAGNETIC BUTTONS ===
document.querySelectorAll(".btn").forEach(btn => {
  btn.addEventListener("mousemove", e => {
    const r = btn.getBoundingClientRect();
    const x = e.clientX - r.left - r.width/2;
    const y = e.clientY - r.top - r.height/2;
    btn.style.transform = `translate(${x*0.25}px, ${y*0.25}px) scale(1.04)`;
  });
  btn.addEventListener("mouseleave", () => { btn.style.transform = ""; });
});

// === COUNTER ANIMATION ===
document.querySelectorAll(".stat-number").forEach(el => {
  const obs = new IntersectionObserver((entries) => {
    if(entries[0].isIntersecting) {
      const text = el.textContent;
      const match = text.match(/([\\d,]+)/);
      if(match) {
        const target = parseInt(match[1].replace(/,/g, ""));
        const suffix = text.replace(match[1], "").trim();
        const prefix = text.substring(0, text.indexOf(match[1]));
        let current = 0;
        const duration = 2000;
        const start = performance.now();
        function step(now) {
          const progress = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          current = Math.floor(target * eased);
          el.textContent = prefix + current.toLocaleString() + suffix;
          if(progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
      }
      obs.unobserve(el);
    }
  }, {threshold:0.5});
  obs.observe(el);
});

// === IMAGE REVEAL ON SCROLL ===
gsap.utils.toArray("img").forEach(img => {
  gsap.from(img, {
    clipPath:"inset(0 100% 0 0)", duration:1.2, ease:"power4.out",
    scrollTrigger:{trigger:img, start:"top 85%"}
  });
});

// === PARALLAX SECTIONS ===
gsap.utils.toArray(".section-dark, .section-light").forEach((section, i) => {
  if(i > 0) {
    gsap.from(section, {
      yPercent:5, ease:"none",
      scrollTrigger:{trigger:section, start:"top bottom", end:"top top", scrub:1}
    });
  }
});

// === TEXT REVEAL FOR HERO ===
const heroH1 = document.querySelector(".hero h1");
if(heroH1 && heroH1.textContent.length < 80) {
  const words = heroH1.textContent.split(" ");
  heroH1.innerHTML = words.map(w => `<span style="display:inline-block;margin-right:0.3em">${w}</span>`).join("");
  gsap.from(heroH1.querySelectorAll("span"), {
    y:40, opacity:0, rotationX:-40, duration:0.8, stagger:0.08, ease:"back.out(1.5)", delay:0.4
  });
}

// === AURORA BLOB MOUSE INTERACTION ===
const blobs = document.querySelectorAll(".aurora-blob");
document.addEventListener("mousemove", e => {
  const x = (e.clientX / window.innerWidth - 0.5) * 30;
  const y = (e.clientY / window.innerHeight - 0.5) * 30;
  blobs.forEach((blob, i) => {
    const factor = (i + 1) * 0.5;
    blob.style.transform = `translate(${x * factor}px, ${y * factor}px)`;
  });
});

// === SECTION GLOW FOLLOW ===
document.querySelectorAll(".section-dark, .section-light").forEach(section => {
  section.addEventListener("mousemove", e => {
    const rect = section.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    section.style.setProperty("--glow-x", x + "px");
    section.style.setProperty("--glow-y", y + "px");
  });
});

// === FLOATING GEOMETRIC SHAPES IN SECTIONS ===
document.querySelectorAll(".section-dark, .section-light").forEach((section, idx) => {
  if(idx % 2 === 0) {
    const shape = document.createElement("div");
    shape.style.cssText = `position:absolute;width:200px;height:200px;border:1px solid rgba(255,255,255,0.03);
      border-radius:${idx % 3 === 0 ? "50%" : "20px"};top:${20+idx*5}%;right:${5+idx*3}%;
      animation:float-shape ${8+idx*2}s ease-in-out infinite;pointer-events:none;z-index:0;
      transform:rotate(${idx*15}deg);`;
    section.appendChild(shape);
  }
});
</script>
</body>
</html>'''

    # === Generate each section via separate API call ===
    system_section = (
        "Generate ONE HTML section for this SPECIFIC business. Output ONLY raw HTML — no markdown fences, no explanation, "
        "no <!DOCTYPE>, no <html>, no <head>, no <style>, no <script>.\n\n"
        "CRITICAL — FILL ALL SPACE. No empty areas. Every section must feel dense and complete:\n"
        "- Every section MUST have images. Use 2-4 images per section minimum.\n"
        "- Use <img> tags with topical URLs: https://images.unsplash.com/photo-[ID]?w=800&h=600&fit=crop "
        "— pick from these IDs based on topic:\n"
        "  Photography: 1542038784456-1ea8e935640e, 1554048612-b6a83d2ed82c, 1471341971476-ae15ff5dd4ea\n"
        "  Food/Restaurant: 1414235077428-338989a2e8c0, 1504674900247-0877df9cc836, 1517248135467-4c7edcad34c4\n"
        "  Tech/SaaS: 1518770660439-4636190af475, 1551288049-bebda4e38f71, 1519389950473-47ba0277781c\n"
        "  Business: 1497366216548-37526070297c, 1553028826-f4804a6dba3b, 1522071820081-009f0129c71c\n"
        "  Fitness: 1534438327276-14e5300c3a48, 1571019613454-1cb2f99b2d8b, 1517836357463-d25dfeac3438\n"
        "  Nature/Wellness: 1506905925346-21bda4d32df4, 1470071459604-3b5ec3a7fe05, 1441974231531-c6227db76b6e\n"
        "  People/Portraits: 1507003211169-0a1dd7228f2d, 1494790108377-be9c29b29330, 1573497019940-1c28c88b4f3e\n"
        "  Creative/Art: 1513364776144-60967b0f800f, 1460661419201-fd4cecdf8a8b, 1547826039-bfc35e0f1ea8\n"
        "  Real Estate: 1600596542815-ffad4c1539a9, 1600585154340-be6161a56a0c, 1512917774080-9991f1c4c750\n"
        "- For ANY other topic, use relevant IDs or fallback to: https://picsum.photos/seed/KEYWORD/800/600\n"
        "- ALWAYS add loading='lazy' and descriptive alt text to images\n\n"
        "OUTPUT RULES:\n"
        "- Output starts with <section or <footer tag and ends with </section> or </footer>\n"
        "- Use the EXACT id and class provided in the request\n"
        "- Wrap content in <div class='container'>\n"
        "- Add a <div class='section-header animate'> with <h2 class='gradient-text'> and <p> for non-hero sections\n"
        "- Hero: <h1> headline, <p> subheadline, <a class='btn btn-primary'> for CTA, <a class='btn btn-outline'> secondary. "
        "Add a large hero image using <img> with relevant Unsplash URL on the right side using a 2-column flex layout.\n"
        "- Use <div class='grid'> for card layouts with 4-6 cards\n"
        "- Each card: <div class='card animate'> with <i class='fas ICON'></i>, <h3>, <p> — write 2-3 sentences per card, not just one\n"
        "- Add images BETWEEN sections content, in cards, next to text blocks, as backgrounds in about/gallery sections\n"
        "- About sections: 2-column with image (left) and text (right) using display:flex;gap:48px;align-items:center\n"
        "- Gallery sections: 3-column grid of images, each with overlay text on hover\n"
        "- Use VARIED animation classes: 'animate', 'animate-left', 'animate-right', 'animate-scale'\n"
        "- Testimonials: 3 cards with <blockquote><p>quote</p></blockquote>, <cite>, and a small <img> avatar (50x50 round)\n"
        "- Stats: 4 stat cards with <div class='stat-number'>NUMBER+</div> and <p>label</p>\n"
        "- Footer: <footer> with <div class='footer-grid'> (4 columns: brand/about, links, links, contact), "
        "<div class='footer-bottom'> with copyright + <div class='social-links'> with fa-brands icons\n"
        "- Use the EXACT copy text provided. Do not invent new copy.\n"
        "- NO EMPTY SPACE. Fill with images, badges, icons, decorative elements, extra text.\n"
        "- Add inline style='display:flex;gap:48px;align-items:center' for 2-column layouts\n"
    )

    section_htmls = []
    for i, section in enumerate(sections):
        sid = section.get("id", f"section-{i}")
        stype = section.get("type", "content")
        theme = "section-dark" if i % 2 == 0 else "section-light"
        if stype == "hero":
            theme = "hero section-dark"
        elif stype == "footer":
            theme = "section-dark"

        section_copy = copy.get(sid, {})
        if not section_copy:
            for key in copy:
                if key.replace("-", "").replace("_", "") == sid.replace("-", "").replace("_", ""):
                    section_copy = copy[key]
                    break

        prompt = json.dumps({
            "section_id": sid,
            "section_type": stype,
            "css_class": theme,
            "copy": section_copy,
            "layout": section.get("layout", ""),
            "content_hints": section.get("content_hints", ""),
            "website_type": intent.get("website_type", ""),
            "business_name": architecture.get("title", ""),
        })

        raw = call_claude(system_section, prompt, max_tokens=3000)
        raw = re.sub(r"^```(?:html)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\n?```\s*$", "", raw)
        section_htmls.append(raw)

    all_sections = "\n\n".join(section_htmls)
    return skeleton_top + all_sections + "\n" + skeleton_bottom


def agent_qa(html, intent):
    """Patch-based QA: returns find/replace fixes."""
    system = (
        "You are a QA engineer. Review this HTML website and return a JSON array of fixes.\n\n"
        "Each fix is an object with:\n"
        '- "find": exact string to find (must be unique)\n'
        '- "replace": the corrected string\n\n'
        "Check for:\n"
        "- Text that would be invisible (same color as background) — fix the color\n"
        "- Missing loading='lazy' on images — add it\n"
        "- Broken icon class names — fix them (fas fa-icon-name format)\n"
        "- Any unclosed tags — close them\n"
        "- Links with empty href — add # or proper link\n\n"
        "If the HTML looks good, return an empty array: []\n"
        "Return ONLY a JSON array. No markdown fences, no explanation."
    )
    prompt = f"Website type: {intent.get('website_type', 'website')}\n\nHTML:\n{html[:8000]}"
    raw = call_claude(system, prompt, max_tokens=2000)

    fixes = parse_json_response(raw, [])
    if not isinstance(fixes, list):
        return html

    for fix in fixes:
        if not isinstance(fix, dict):
            continue
        find = fix.get("find", "")
        replace = fix.get("replace", "")
        if find and replace and find in html:
            html = html.replace(find, replace, 1)

    return html


def inject_user_images(html, images):
    """Replace placeholder image src attributes with user-uploaded base64 images."""
    if not images:
        return html

    img_pattern = re.compile(r'<img([^>]*?)src=["\']([^"\']*?(?:picsum\.photos|unsplash\.com)[^"\']*?)["\']', re.IGNORECASE)
    matches = list(img_pattern.finditer(html))

    if not matches:
        return html

    replacements = []
    for i, match in enumerate(matches):
        if i >= len(images):
            break
        old_src = match.group(2)
        new_src = images[i]
        replacements.append((old_src, new_src))

    for old_src, new_src in replacements:
        html = html.replace(old_src, new_src, 1)

    return html


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "echo-websites"})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    if not data.get("prompt") and not data.get("business_name") and not data.get("website_type"):
        return jsonify({"error": "Please provide a description, business name, or website type"}), 400

    enhanced_prompt = build_enhanced_prompt(data)
    user_images = data.get("images", [])

    def stream():
        try:
            yield json.dumps({"stage": "intent", "message": "Analyzing your request..."}) + "\n"
            intent = agent_intent_analyzer(enhanced_prompt)
            yield json.dumps({"stage": "intent_done", "message": f"Building a {intent.get('website_type', 'custom')} website"}) + "\n"

            yield json.dumps({"stage": "architect", "message": "Designing page structure..."}) + "\n"
            architecture = agent_site_architect(intent)
            section_count = len(architecture.get("sections", []))
            yield json.dumps({"stage": "architect_done", "message": f"Planned {section_count} sections"}) + "\n"

            yield json.dumps({"stage": "design", "message": "Creating visual identity..."}) + "\n"
            design = agent_design(intent, architecture)
            yield json.dumps({"stage": "design_done", "message": f"{design.get('style_notes', 'custom design')[:60]}"}) + "\n"

            yield json.dumps({"stage": "copy", "message": "Writing content..."}) + "\n"
            copy = agent_copywriter(intent, architecture)
            yield json.dumps({"stage": "copy_done", "message": "Content written"}) + "\n"

            section_count = len(architecture.get("sections", []))
            yield json.dumps({"stage": "build", "message": f"Building {section_count} sections with 3D effects..."}) + "\n"
            html = agent_component_builder(intent, architecture, design, copy)
            yield json.dumps({"stage": "build_done", "message": f"Built {section_count} interactive sections"}) + "\n"

            yield json.dumps({"stage": "qa", "message": "Running quality checks..."}) + "\n"
            final_html = agent_qa(html, intent)

            # Inject user-uploaded images if provided
            if user_images:
                final_html = inject_user_images(final_html, user_images)

            yield json.dumps({"stage": "qa_done", "message": "Quality verified"}) + "\n"

            yield json.dumps({"stage": "complete", "html": final_html}) + "\n"

        except Exception as e:
            yield json.dumps({"stage": "error", "message": str(e)}) + "\n"

    return Response(
        stream_with_context(stream()),
        mimetype="text/plain",
        headers={"X-Content-Type-Options": "nosniff"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
