import os
import re
import json
import time
import threading
import anthropic
from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
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


def call_claude_stream(system_prompt, user_prompt, max_tokens=5000):
    """Streamed API call — keeps the connection alive while tokens generate."""
    chunks = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
    return "".join(chunks)


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


def build_enhanced_prompt(data):
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
            "portfolio": "Portfolio / Showcase", "landing-page": "Landing Page",
            "restaurant": "Restaurant / Food & Drink", "business": "Business / Corporate",
            "agency": "Creative Agency", "saas": "SaaS / Software Product",
            "ecommerce": "E-commerce / Online Store", "blog": "Blog / Magazine",
            "event": "Event / Conference", "personal": "Personal / Resume",
            "fitness": "Fitness / Wellness", "real-estate": "Real Estate",
        }
        parts.append(f"Website type: {type_labels.get(website_type, website_type)}")
    if style:
        style_descriptions = {
            "professional": "Professional — clean, corporate, trustworthy",
            "creative": "Creative — artistic, expressive, bold visual choices",
            "minimal": "Minimal — whitespace, simple, elegant",
            "bold": "Bold — high contrast, large typography, energetic",
            "luxury": "Luxury — premium, elegant, dark tones with rich accents",
            "tech": "Tech — futuristic, neon on dark, geometric",
        }
        parts.append(f"Design style: {style_descriptions.get(style, style)}")
    if color_preference and color_preference != "auto":
        palette_descriptions = {
            "midnight": "Dark purples and deep blues",
            "ocean": "Deep blues and teals",
            "ember": "Warm reds and oranges on dark",
            "forest": "Greens and earthy tones",
            "rose": "Pinks and magentas",
            "mono": "Black, white, and grays",
        }
        parts.append(f"Color palette: {palette_descriptions.get(color_preference, color_preference)}")
    if features:
        feature_labels = {
            "contact": "Contact form", "gallery": "Image gallery",
            "testimonials": "Testimonials", "pricing": "Pricing table",
            "faq": "FAQ", "team": "Team grid",
            "newsletter": "Newsletter signup", "stats": "Statistics",
        }
        feature_list = [feature_labels.get(f, f) for f in features]
        parts.append(f"MUST include: {', '.join(feature_list)}")
    return "\n".join(parts)


# ─── CALL 1: Plan (intent + architecture + design) ───────────────────────────

def agent_plan(user_prompt):
    system = (
        "You are a senior creative strategist, UX architect, and visual designer.\n"
        "Given a website request, produce a SINGLE JSON object with three top-level keys: "
        '"intent", "architecture", and "design".\n\n'
        "CRITICAL: Follow the user's request EXACTLY. If they say 'photographer portfolio', "
        "everything must be photographer-specific. Do NOT generalize.\n\n"
        '=== "intent" ===\n'
        '- "website_type": string\n'
        '- "audience": string\n'
        '- "goal": string\n'
        '- "tone": string\n'
        '- "key_features": list of 8-12 strings specific to this business\n'
        '- "industry": string\n'
        '- "design_inspiration": string\n'
        '- "emotional_response": string\n\n'
        '=== "architecture" ===\n'
        '- "title": string (business name or memorable title)\n'
        '- "nav_items": list of 4-6 short strings\n'
        '- "sections": list of 7-10 objects, each with:\n'
        '  "id" (html id), "type" (hero/about/gallery/services/testimonials/contact/footer/features/pricing/cta/stats/faq/team/portfolio/process/menu/case-studies),\n'
        '  "purpose", "layout" (e.g. "3-card grid", "2-col image+text"), "content_hints", "visual_effect"\n'
        "Hero first, footer last. Sections specific to the business type.\n\n"
        '=== "design" ===\n'
        "All hex colors: primary_color, primary_light, primary_dark, secondary_color, accent_color,\n"
        "background_color, background_alt, surface_color, text_color, text_secondary, text_on_primary,\n"
        "gradient_start, gradient_end, gradient_angle (string).\n"
        "Typography: heading_font, body_font (Google Fonts), hero_size, h2_size, body_size, heading_weight, heading_letter_spacing.\n"
        "Spacing: border_radius, section_padding, card_padding, max_width.\n"
        "Effects: shadow_sm, shadow_lg (CSS strings), glass_effect (bool), use_gradients (bool), style_notes (string).\n\n"
        "Return ONLY valid JSON, no markdown fences."
    )
    raw = call_claude(system, user_prompt, max_tokens=4096)
    fallback = {
        "intent": {
            "website_type": "website", "audience": "general", "goal": "inform visitors",
            "tone": "professional",
            "key_features": ["hero", "about", "services", "testimonials", "contact", "footer"],
            "industry": "general", "design_inspiration": "clean modern website",
            "emotional_response": "trust and interest",
        },
        "architecture": {
            "title": "My Website", "nav_items": ["Home", "About", "Services", "Contact"],
            "sections": [
                {"id": "hero", "type": "hero", "purpose": "First impression", "layout": "centered", "content_hints": "Headline and CTA", "visual_effect": "fade-in"},
                {"id": "about", "type": "about", "purpose": "Story", "layout": "2-column", "content_hints": "Bio and image", "visual_effect": "fade-in-up"},
                {"id": "services", "type": "services", "purpose": "Offerings", "layout": "3-card grid", "content_hints": "3-4 cards", "visual_effect": "stagger"},
                {"id": "testimonials", "type": "testimonials", "purpose": "Social proof", "layout": "3-card grid", "content_hints": "3 quotes", "visual_effect": "fade-in"},
                {"id": "contact", "type": "contact", "purpose": "Conversion", "layout": "centered", "content_hints": "CTA", "visual_effect": "fade-in-up"},
                {"id": "footer", "type": "footer", "purpose": "Nav and info", "layout": "3-column", "content_hints": "Links", "visual_effect": "none"},
            ],
        },
        "design": {
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
        },
    }
    result = parse_json_response(raw, fallback)
    return (
        result.get("intent", fallback["intent"]),
        result.get("architecture", fallback["architecture"]),
        result.get("design", fallback["design"]),
    )


# ─── CALL 2: Component Builder (streamed, single call for all sections) ───────

def agent_component_builder(intent, architecture, design):
    """Hardcoded skeleton + ONE streamed API call for all section HTML."""

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

    # ── Hardcoded skeleton: head, CSS, nav, JS ──
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
  --primary:{primary};--primary-light:{primary_light};--primary-dark:{primary_dark};
  --secondary:{secondary};--accent:{accent};
  --bg:{bg};--bg-alt:{bg_alt};--surface:{surface};
  --text:{text_color};--text-sec:{text_sec};--text-on-primary:{text_on_primary};
  --grad-start:{grad_start};--grad-end:{grad_end};--grad-angle:{grad_angle};
  --radius:{radius};--shadow-sm:{shadow_sm};--shadow-lg:{shadow_lg};
  --heading-font:'{heading_font}',sans-serif;--body-font:'{body_font}',sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{font-family:var(--body-font);color:var(--text);background:var(--bg);line-height:1.6;overflow-x:hidden}}
.container{{max-width:1200px;margin:0 auto;padding:0 24px;position:relative}}

/* Loader */
.page-loader{{position:fixed;inset:0;background:var(--bg);display:flex;align-items:center;justify-content:center;z-index:99999;transition:opacity .6s,visibility .6s}}
.page-loader.hidden{{opacity:0;visibility:hidden;pointer-events:none}}
.loader-ring{{width:40px;height:40px;border:3px solid rgba(255,255,255,.1);border-top-color:var(--primary);border-radius:50%;animation:spin .8s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}

/* Scroll progress */
.scroll-progress{{position:fixed;top:0;left:0;height:3px;background:linear-gradient(90deg,var(--grad-start),var(--grad-end));z-index:10001;width:0;transition:width .1s}}

/* Cursor */
.cursor-glow{{position:fixed;width:24px;height:24px;border-radius:50%;pointer-events:none;z-index:9999;background:radial-gradient(circle,var(--primary),transparent 70%);opacity:.4;transform:translate(-50%,-50%);transition:width .3s,height .3s,opacity .3s;mix-blend-mode:screen}}
.cursor-glow.hover{{width:48px;height:48px;opacity:.6}}
@media(max-width:768px){{.cursor-glow{{display:none}}}}

/* Backgrounds */
body::before{{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.03;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}}
.aurora{{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}}
.aurora-blob{{position:absolute;border-radius:50%;filter:blur(80px);opacity:.12;animation:aurora-drift 12s ease-in-out infinite alternate}}
.aurora-blob:nth-child(1){{width:600px;height:600px;top:-200px;left:-100px;background:var(--primary);animation-duration:14s}}
.aurora-blob:nth-child(2){{width:500px;height:500px;bottom:-150px;right:-100px;background:var(--grad-end);animation-delay:3s;animation-duration:16s}}
.aurora-blob:nth-child(3){{width:400px;height:400px;top:40%;left:50%;background:var(--secondary);animation-delay:6s;animation-duration:18s}}
@keyframes aurora-drift{{0%{{transform:translate(0,0) scale(1)}}50%{{transform:translate(40px,-30px) scale(1.1)}}100%{{transform:translate(-20px,20px) scale(.95)}}}}
.particle-canvas{{position:fixed;inset:0;z-index:0;pointer-events:none}}

/* Section themes */
.section-dark{{background:var(--bg);color:#fff;padding:90px 0;position:relative;overflow:hidden}}
.section-dark::before{{content:'';position:absolute;inset:0;pointer-events:none;background-image:radial-gradient(rgba(255,255,255,.03) 1px,transparent 1px);background-size:32px 32px;z-index:0}}
.section-dark::after{{content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);width:800px;height:400px;pointer-events:none;background:radial-gradient(ellipse,var(--primary),transparent 70%);opacity:.04;z-index:0}}
.section-dark>*{{position:relative;z-index:1}}
.section-dark h1,.section-dark h2,.section-dark h3{{color:#fff;font-family:var(--heading-font)}}
.section-dark p,.section-dark li,.section-dark span{{color:rgba(255,255,255,.85)}}
.section-dark .card{{background:var(--surface);color:#fff;border:1px solid rgba(255,255,255,.08);backdrop-filter:blur(10px)}}
.section-dark .card h3{{color:#fff}}.section-dark .card p{{color:rgba(255,255,255,.8)}}

.section-light{{background:var(--bg-alt);color:#fff;padding:90px 0;position:relative;overflow:hidden}}
.section-light::before{{content:'';position:absolute;inset:0;pointer-events:none;background-image:radial-gradient(rgba(255,255,255,.04) 1px,transparent 1px);background-size:32px 32px;z-index:0}}
.section-light::after{{content:'';position:absolute;bottom:0;right:10%;width:600px;height:300px;pointer-events:none;background:radial-gradient(ellipse,var(--grad-end),transparent 70%);opacity:.05;z-index:0}}
.section-light>*{{position:relative;z-index:1}}
.section-light h1,.section-light h2,.section-light h3{{color:#fff;font-family:var(--heading-font)}}
.section-light p,.section-light li,.section-light span{{color:rgba(255,255,255,.8)}}
.section-light .card{{background:rgba(255,255,255,.03);color:#fff;border:1px solid rgba(255,255,255,.08);backdrop-filter:blur(10px)}}
.section-light .card h3{{color:#fff}}.section-light .card p{{color:rgba(255,255,255,.75)}}

/* Hero */
.hero{{min-height:100vh;display:flex;align-items:center;position:relative;overflow:hidden;background:linear-gradient(var(--grad-angle),var(--grad-start),var(--grad-end))}}
.hero::before{{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 20% 50%,rgba(255,255,255,.15) 0%,transparent 50%),radial-gradient(ellipse at 80% 20%,rgba(255,255,255,.08) 0%,transparent 40%);opacity:.6;animation:hero-bg 10s ease-in-out infinite alternate}}
@keyframes hero-bg{{0%{{transform:scale(1);opacity:.6}}100%{{transform:scale(1.1) translate(2%,-2%);opacity:.4}}}}
.hero::after{{content:'';position:absolute;inset:0;background:rgba(0,0,0,.15);background-image:radial-gradient(rgba(255,255,255,.02) 1px,transparent 1px);background-size:24px 24px}}
.hero .container{{position:relative;z-index:2}}
.hero h1{{font-size:clamp(2.8rem,7vw,5.5rem);font-weight:900;line-height:1.05;margin-bottom:24px;color:#fff;letter-spacing:-.03em}}
.hero p{{font-size:clamp(1.05rem,2vw,1.3rem);margin-bottom:36px;color:rgba(255,255,255,.85);max-width:600px;line-height:1.7}}
.hero .btn{{margin-right:12px;margin-bottom:12px}}
.hero-shapes{{position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:1}}
.hero-shape{{position:absolute;border-radius:50%;opacity:.08;background:rgba(255,255,255,.5);animation:float-shape 8s ease-in-out infinite}}
.hero-shape:nth-child(1){{width:300px;height:300px;top:-50px;right:-50px}}
.hero-shape:nth-child(2){{width:200px;height:200px;bottom:10%;left:5%;animation-delay:2s}}
.hero-shape:nth-child(3){{width:150px;height:150px;top:20%;right:20%;animation-delay:4s}}
.hero-shape:nth-child(4){{width:100px;height:100px;bottom:30%;right:10%;animation-delay:1s}}
@keyframes float-shape{{0%,100%{{transform:translateY(0) scale(1)}}33%{{transform:translateY(-20px) scale(1.05)}}66%{{transform:translateY(10px) scale(.95)}}}}

/* Typography */
h2{{font-size:clamp(2rem,4.5vw,3.2rem);font-weight:800;margin-bottom:16px;font-family:var(--heading-font);letter-spacing:-.02em;line-height:1.15}}
h3{{font-size:1.3rem;font-weight:600;margin-bottom:10px;font-family:var(--heading-font)}}
.section-header{{text-align:center;max-width:700px;margin:0 auto 48px}}
.section-header p{{font-size:1.1rem;line-height:1.7}}
.gradient-text{{background:linear-gradient(var(--grad-angle),var(--grad-start),var(--grad-end));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}

/* Grid & Cards */
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:32px}}
.card{{padding:36px;border-radius:var(--radius);transition:all .4s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden}}
.card::before{{content:'';position:absolute;inset:0;border-radius:inherit;opacity:0;background:linear-gradient(var(--grad-angle),var(--primary),transparent);transition:opacity .4s;z-index:0}}
.card:hover::before{{opacity:.05}}
.card:hover{{transform:translateY(-12px) scale(1.02);box-shadow:var(--shadow-lg),0 0 40px rgba(0,0,0,.1)}}
.card>*{{position:relative;z-index:1}}
.card i{{font-size:2.2rem;margin-bottom:18px;display:block;background:linear-gradient(var(--grad-angle),var(--grad-start),var(--grad-end));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}

/* Buttons */
.btn{{display:inline-flex;align-items:center;gap:8px;padding:16px 36px;border-radius:8px;font-weight:600;text-decoration:none;transition:all .3s cubic-bezier(.4,0,.2,1);cursor:pointer;border:none;font-size:1rem;font-family:var(--body-font);position:relative;overflow:hidden}}
.btn-primary{{background:linear-gradient(135deg,var(--primary),var(--primary-dark));color:var(--text-on-primary);box-shadow:0 4px 20px rgba(0,0,0,.2)}}
.btn-primary:hover{{transform:translateY(-3px) scale(1.03);box-shadow:0 8px 30px rgba(0,0,0,.3);filter:brightness(1.1)}}
.btn-primary::after{{content:'';position:absolute;inset:-50%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent);transform:rotate(45deg) translateX(-100%);transition:transform .6s}}
.btn-primary:hover::after{{transform:rotate(45deg) translateX(100%)}}
.btn-outline{{background:transparent;border:2px solid rgba(255,255,255,.25);color:#fff}}
.btn-outline:hover{{background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.4);transform:translateY(-3px)}}

/* Images */
img{{max-width:100%;height:auto;border-radius:var(--radius);display:block}}
.img-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}}
.img-grid img{{width:100%;height:250px;object-fit:cover;transition:transform .4s,filter .4s}}
.img-grid img:hover{{transform:scale(1.03);filter:brightness(1.1)}}
.two-col{{display:flex;gap:48px;align-items:center}}
.two-col>*{{flex:1}}
@media(max-width:768px){{.two-col{{flex-direction:column;gap:32px}}}}
.badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.75rem;font-weight:600;background:rgba(124,92,252,.1);color:var(--primary);border:1px solid rgba(124,92,252,.2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}}
.avatar{{width:48px;height:48px;border-radius:50%;object-fit:cover;border:2px solid var(--primary)}}
.testimonial-author{{display:flex;align-items:center;gap:12px;margin-top:16px}}
.img-overlay-wrap{{position:relative;overflow:hidden;border-radius:var(--radius)}}
.img-overlay-wrap img{{width:100%;height:300px;object-fit:cover;transition:transform .5s}}
.img-overlay-wrap:hover img{{transform:scale(1.08)}}
.img-overlay{{position:absolute;inset:0;background:linear-gradient(to top,rgba(0,0,0,.7),transparent);display:flex;align-items:flex-end;padding:20px;opacity:0;transition:opacity .4s}}
.img-overlay-wrap:hover .img-overlay{{opacity:1}}
.img-overlay p{{color:#fff;font-weight:600}}

/* Animations */
.animate{{opacity:0;transform:translateY(40px);transition:opacity .7s cubic-bezier(.4,0,.2,1),transform .7s cubic-bezier(.4,0,.2,1)}}
.animate.visible{{opacity:1;transform:translateY(0)}}
.animate-left{{opacity:0;transform:translateX(-40px);transition:opacity .7s ease,transform .7s ease}}
.animate-left.visible{{opacity:1;transform:translateX(0)}}
.animate-right{{opacity:0;transform:translateX(40px);transition:opacity .7s ease,transform .7s ease}}
.animate-right.visible{{opacity:1;transform:translateX(0)}}
.animate-scale{{opacity:0;transform:scale(.9);transition:opacity .7s ease,transform .7s ease}}
.animate-scale.visible{{opacity:1;transform:scale(1)}}

/* Nav */
header{{position:fixed;top:0;left:0;right:0;z-index:1000;padding:18px 0;background:transparent;transition:all .4s cubic-bezier(.4,0,.2,1)}}
header.scrolled{{background:rgba({int(bg[1:3],16)},{int(bg[3:5],16)},{int(bg[5:7],16)},.92);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);box-shadow:0 2px 30px rgba(0,0,0,.3);padding:12px 0}}
nav{{display:flex;align-items:center;justify-content:space-between}}
.logo{{font-family:var(--heading-font);font-size:1.5rem;font-weight:800;color:#fff;text-decoration:none;letter-spacing:-.02em}}
.nav-links{{display:flex;gap:28px;list-style:none}}
.nav-links a{{color:rgba(255,255,255,.75);text-decoration:none;font-weight:500;font-size:.95rem;transition:color .3s;position:relative}}
.nav-links a:hover{{color:#fff}}
.nav-links a::after{{content:'';position:absolute;bottom:-4px;left:0;width:0;height:2px;background:var(--primary);transition:width .3s;border-radius:2px}}
.nav-links a:hover::after{{width:100%}}
.hamburger{{display:none;flex-direction:column;gap:5px;cursor:pointer;background:none;border:none;padding:8px}}
.hamburger span{{width:24px;height:2px;background:#fff;transition:all .3s;border-radius:2px}}

/* Footer */
footer{{background:var(--bg);color:rgba(255,255,255,.7);padding:80px 0 40px}}
footer h3{{color:#fff;margin-bottom:18px;font-size:1.1rem}}
footer a{{color:rgba(255,255,255,.5);text-decoration:none;transition:color .3s}}
footer a:hover{{color:var(--primary-light)}}
.footer-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:48px}}
.footer-bottom{{border-top:1px solid rgba(255,255,255,.08);margin-top:48px;padding-top:28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px}}
.social-links{{display:flex;gap:18px}}
.social-links a{{font-size:1.3rem;color:rgba(255,255,255,.5);transition:all .3s}}
.social-links a:hover{{color:var(--primary-light);transform:translateY(-3px)}}
blockquote{{font-style:italic;font-size:1.1rem;line-height:1.8;margin-bottom:18px;position:relative;padding-left:20px;border-left:3px solid var(--primary)}}
cite{{font-style:normal;font-weight:600;font-size:.95rem}}
.stat-number{{font-size:clamp(2.5rem,5vw,4rem);font-weight:900;color:var(--primary);font-family:var(--heading-font);line-height:1}}

/* Responsive */
@media(max-width:768px){{
  .grid{{grid-template-columns:1fr}}
  section,.section-dark,.section-light{{padding:60px 0}}
  .nav-links{{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba({int(bg[1:3],16)},{int(bg[3:5],16)},{int(bg[5:7],16)},.98);flex-direction:column;align-items:center;justify-content:center;gap:36px;z-index:999;backdrop-filter:blur(20px)}}
  .nav-links.active{{display:flex}}.nav-links a{{font-size:1.3rem}}
  .hamburger{{display:flex;z-index:1000}}
  .footer-grid{{grid-template-columns:1fr}}.footer-bottom{{flex-direction:column;text-align:center}}
  .hero h1{{font-size:clamp(2rem,8vw,3.5rem)}}
}}
@media(prefers-reduced-motion:reduce){{.animate,.animate-left,.animate-right,.animate-scale{{opacity:1;transform:none;transition:none}}.hero-shape{{animation:none}}.cursor-glow{{display:none}}}}
</style>
</head>
<body>
<div class="aurora"><div class="aurora-blob"></div><div class="aurora-blob"></div><div class="aurora-blob"></div></div>
<canvas class="particle-canvas" id="particleCanvas"></canvas>
<div class="page-loader" id="pageLoader"><div class="loader-ring"></div></div>
<div class="scroll-progress" id="scrollProgress"></div>
<div class="cursor-glow" id="cursorGlow"></div>
<header><div class="container"><nav>
<a href="#" class="logo">{title}</a>
<div class="nav-links">{nav_html}</div>
<button class="hamburger" aria-label="Menu"><span></span><span></span><span></span></button>
</nav></div></header>
<main>
'''

    skeleton_bottom = '''</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js"></script>
<script>
gsap.registerPlugin(ScrollTrigger);
// Particles
(function(){const c=document.getElementById("particleCanvas");if(!c)return;const x=c.getContext("2d");let p=[];const N=50;function r(){c.width=innerWidth;c.height=innerHeight}r();addEventListener("resize",r);for(let i=0;i<N;i++)p.push({x:Math.random()*c.width,y:Math.random()*c.height,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,s:Math.random()*2+.5,o:Math.random()*.4+.1});function d(){x.clearRect(0,0,c.width,c.height);const cl=getComputedStyle(document.documentElement).getPropertyValue("--primary").trim()||"#6c5ce7";p.forEach((a,i)=>{a.x+=a.vx;a.y+=a.vy;if(a.x<0||a.x>c.width)a.vx*=-1;if(a.y<0||a.y>c.height)a.vy*=-1;x.beginPath();x.arc(a.x,a.y,a.s,0,Math.PI*2);x.fillStyle=cl;x.globalAlpha=a.o;x.fill();for(let j=i+1;j<p.length;j++){const dx=a.x-p[j].x,dy=a.y-p[j].y,dt=Math.sqrt(dx*dx+dy*dy);if(dt<120){x.beginPath();x.moveTo(a.x,a.y);x.lineTo(p[j].x,p[j].y);x.strokeStyle=cl;x.globalAlpha=(1-dt/120)*.08;x.lineWidth=.5;x.stroke()}}});x.globalAlpha=1;requestAnimationFrame(d)}d()})();
// Loader
addEventListener("load",()=>{setTimeout(()=>{document.getElementById("pageLoader").classList.add("hidden");gsap.from(".hero h1",{y:60,opacity:0,duration:1,delay:.2,ease:"power3.out"});gsap.from(".hero p",{y:40,opacity:0,duration:.8,delay:.5,ease:"power3.out"});gsap.from(".hero .btn",{y:30,opacity:0,duration:.6,delay:.8,stagger:.15,ease:"power3.out"})},300)});
// Scroll progress
addEventListener("scroll",()=>{document.getElementById("scrollProgress").style.width=Math.min(scrollY/(document.documentElement.scrollHeight-innerHeight)*100,100)+"%"});
// Cursor
const cg=document.getElementById("cursorGlow");let mx=0,my=0,cx=0,cy=0;document.addEventListener("mousemove",e=>{mx=e.clientX;my=e.clientY});(function ac(){cx+=(mx-cx)*.15;cy+=(my-cy)*.15;cg.style.left=cx+"px";cg.style.top=cy+"px";requestAnimationFrame(ac)})();document.querySelectorAll("a,.btn,.card").forEach(e=>{e.addEventListener("mouseenter",()=>cg.classList.add("hover"));e.addEventListener("mouseleave",()=>cg.classList.remove("hover"))});
// Scroll reveal
const obs=new IntersectionObserver(e=>{e.forEach(t=>{if(t.isIntersecting){t.target.classList.add("visible");obs.unobserve(t.target)}})},{threshold:.1,rootMargin:"0px 0px -60px 0px"});document.querySelectorAll(".animate,.animate-left,.animate-right,.animate-scale").forEach(e=>obs.observe(e));
// GSAP headings
gsap.utils.toArray("section h2,.section-dark h2,.section-light h2").forEach(h=>{gsap.from(h,{y:60,opacity:0,duration:.9,ease:"power3.out",scrollTrigger:{trigger:h,start:"top 85%"}})});
// Card stagger
gsap.utils.toArray(".grid").forEach(g=>{const c=g.querySelectorAll(".card");if(c.length)gsap.from(c,{y:80,opacity:0,duration:.7,stagger:.12,ease:"power3.out",scrollTrigger:{trigger:g,start:"top 82%"}})});
// Hero parallax
if(document.querySelector(".hero")){gsap.to(".hero",{yPercent:-20,ease:"none",scrollTrigger:{trigger:".hero",scrub:1.5}})}
// Sticky nav
addEventListener("scroll",()=>{const h=document.querySelector("header");if(h)h.classList.toggle("scrolled",scrollY>80)});
// Mobile menu
const hb=document.querySelector(".hamburger"),nl=document.querySelector(".nav-links");if(hb&&nl)hb.addEventListener("click",()=>{nl.classList.toggle("active");hb.classList.toggle("active")});
// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a=>{a.addEventListener("click",e=>{e.preventDefault();const t=document.querySelector(a.getAttribute("href"));if(t)t.scrollIntoView({behavior:"smooth"});if(nl)nl.classList.remove("active")})});
// 3D card tilt
document.querySelectorAll(".card").forEach(c=>{c.addEventListener("mousemove",e=>{const r=c.getBoundingClientRect(),x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;c.style.transform=`perspective(1000px) rotateY(${x*12}deg) rotateX(${y*-12}deg) translateY(-12px) scale(1.02)`;c.style.transition="transform .1s"});c.addEventListener("mouseleave",()=>{c.style.transform="";c.style.transition="transform .5s cubic-bezier(.4,0,.2,1)"})});
// Magnetic buttons
document.querySelectorAll(".btn").forEach(b=>{b.addEventListener("mousemove",e=>{const r=b.getBoundingClientRect(),x=e.clientX-r.left-r.width/2,y=e.clientY-r.top-r.height/2;b.style.transform=`translate(${x*.25}px,${y*.25}px) scale(1.04)`});b.addEventListener("mouseleave",()=>{b.style.transform=""})});
// Counter animation
document.querySelectorAll(".stat-number").forEach(el=>{const o=new IntersectionObserver(e=>{if(e[0].isIntersecting){const t=el.textContent,m=t.match(/([\\d,]+)/);if(m){const tgt=parseInt(m[1].replace(/,/g,"")),sfx=t.replace(m[1],"").trim(),pfx=t.substring(0,t.indexOf(m[1]));let s=performance.now();(function step(n){const p=Math.min((n-s)/2000,1),v=Math.floor(tgt*(1-Math.pow(1-p,3)));el.textContent=pfx+v.toLocaleString()+sfx;if(p<1)requestAnimationFrame(step)})(s)}o.unobserve(el)}},{threshold:.5});o.observe(el)});
// Image reveal
gsap.utils.toArray("img").forEach(i=>{gsap.from(i,{clipPath:"inset(0 100% 0 0)",duration:1.2,ease:"power4.out",scrollTrigger:{trigger:i,start:"top 85%"}})});
// Aurora mouse
const blobs=document.querySelectorAll(".aurora-blob");document.addEventListener("mousemove",e=>{const x=(e.clientX/innerWidth-.5)*30,y=(e.clientY/innerHeight-.5)*30;blobs.forEach((b,i)=>{b.style.transform=`translate(${x*(i+1)*.5}px,${y*(i+1)*.5}px)`})});
// Hero text reveal
const hh=document.querySelector(".hero h1");if(hh&&hh.textContent.length<80){const w=hh.textContent.split(" ");hh.innerHTML=w.map(s=>`<span style="display:inline-block;margin-right:.3em">${s}</span>`).join("");gsap.from(hh.querySelectorAll("span"),{y:40,opacity:0,rotationX:-40,duration:.8,stagger:.08,ease:"back.out(1.5)",delay:.4})}
</script>
</body>
</html>'''

    # ── Build section spec for the single API call ──
    sections_spec = []
    for i, section in enumerate(sections):
        stype = section.get("type", "content")
        theme = "section-dark" if i % 2 == 0 else "section-light"
        if stype == "hero":
            theme = "hero section-dark"
        elif stype == "footer":
            theme = "section-dark"
        sections_spec.append({
            "id": section.get("id", f"s{i}"),
            "type": stype,
            "class": theme,
            "layout": section.get("layout", ""),
            "hints": section.get("content_hints", ""),
            "purpose": section.get("purpose", ""),
        })

    system_build = (
        "You are an expert frontend developer. Generate ALL section HTML for a complete website.\n"
        "Output ONLY raw HTML — no markdown fences, no doctype/head/style/script.\n"
        "Write vivid, specific copy inline — tailored to this exact business. NEVER use generic filler.\n\n"
        "AVAILABLE CSS CLASSES (already defined — just use them):\n"
        "Layout: container, grid, two-col, section-header, img-grid\n"
        "Cards: card animate (with i.fas.fa-icon, h3, p inside)\n"
        "Buttons: btn btn-primary, btn btn-outline\n"
        "Text: gradient-text, badge, stat-number\n"
        "Animation: animate, animate-left, animate-right, animate-scale\n"
        "Images: avatar, img-overlay-wrap + img-overlay\n"
        "Testimonials: blockquote > p, cite, testimonial-author\n"
        "Footer: footer-grid, footer-bottom, social-links\n\n"
        "RULES:\n"
        "- Each section: <section id=\"ID\" class=\"CLASS\"><div class=\"container\">...</div></section>\n"
        "- Footer: <footer class=\"section-dark\"><div class=\"container\">...</div></footer>\n"
        "- Hero: h1 + p + buttons. Add <div class='hero-shapes'><div class='hero-shape'></div><div class='hero-shape'></div><div class='hero-shape'></div><div class='hero-shape'></div></div>\n"
        "- Non-hero sections: start with <div class='section-header animate'><h2 class='gradient-text'>headline</h2><p>subtitle</p></div>\n"
        "- Use Font Awesome 6: <i class='fas fa-icon-name'></i>\n"
        "- Images: <img src='https://picsum.photos/seed/KEYWORD/800/600' loading='lazy' alt='...'>\n"
        "- Include images in hero, about, gallery, and other visual sections\n"
        "- Cards: 4-6 per grid with icon + h3 + 2-sentence p\n"
        "- Testimonials: 3 cards with blockquote + cite + avatar\n"
        "- Stats: 4 cards with stat-number + label\n"
        "- Footer: 4-col footer-grid + footer-bottom with social-links (fa-brands icons)\n"
        "- Write efficient HTML — use the classes above, avoid inline styles when a class works\n"
        "- Be DENSE — no empty space, fill every section with rich content\n"
        "- Write ALL sections listed below, in order, one after another\n"
    )

    user_prompt = json.dumps({
        "website_type": intent.get("website_type", "website"),
        "business_name": title,
        "tone": intent.get("tone", "professional"),
        "industry": intent.get("industry", "general"),
        "sections": sections_spec,
    })

    raw = call_claude_stream(system_build, user_prompt, max_tokens=5000)
    raw = re.sub(r"^```(?:html)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)

    return skeleton_top + raw + "\n" + skeleton_bottom


# ─── Image injection ──────────────────────────────────────────────────────────

def inject_user_images(html, images):
    if not images:
        return html
    img_pattern = re.compile(r'<img([^>]*?)src=["\']([^"\']*?(?:picsum\.photos|unsplash\.com)[^"\']*?)["\']', re.IGNORECASE)
    matches = list(img_pattern.finditer(html))
    if not matches:
        return html
    for i, match in enumerate(matches):
        if i >= len(images):
            break
        html = html.replace(match.group(2), images[i], 1)
    return html


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


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

    def sse(data):
        return f"data: {json.dumps(data)}\n\n"

    def stream():
        try:
            yield sse({"stage": "plan", "message": "Planning intent, structure & design..."})

            plan_result = {"intent": None, "architecture": None, "design": None, "error": None}
            plan_done = threading.Event()

            def _plan_worker():
                try:
                    i, a, d = agent_plan(enhanced_prompt)
                    plan_result["intent"] = i
                    plan_result["architecture"] = a
                    plan_result["design"] = d
                except Exception as e:
                    plan_result["error"] = e
                finally:
                    plan_done.set()

            threading.Thread(target=_plan_worker, daemon=True).start()
            elapsed = 0
            while not plan_done.wait(timeout=10):
                elapsed += 10
                yield sse({"stage": "ping", "message": f"Planning... ({elapsed}s)"})

            if plan_result["error"] is not None:
                raise plan_result["error"]

            intent = plan_result["intent"] or {}
            architecture = plan_result["architecture"] or {}
            design = plan_result["design"] or {}
            section_count = len(architecture.get("sections", []))
            yield sse({"stage": "plan_done", "message": f"{intent.get('website_type', 'custom')} — {section_count} sections planned"})

            yield sse({"stage": "build", "message": f"Building {section_count} sections with 3D effects..."})

            build_result = {"html": None, "error": None}
            build_done = threading.Event()

            def _build_worker():
                try:
                    build_result["html"] = agent_component_builder(intent, architecture, design)
                except Exception as e:
                    build_result["error"] = e
                finally:
                    build_done.set()

            threading.Thread(target=_build_worker, daemon=True).start()
            elapsed = 0
            while not build_done.wait(timeout=10):
                elapsed += 10
                yield sse({"stage": "ping", "message": f"Building sections... ({elapsed}s)"})

            if build_result["error"] is not None:
                raise build_result["error"]

            html = build_result["html"] or ""
            yield sse({"stage": "build_done", "message": f"Built {section_count} interactive sections"})

            if user_images:
                html = inject_user_images(html, user_images)

            yield sse({"stage": "complete", "html": html})

        except Exception as e:
            yield sse({"stage": "error", "message": str(e)})

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
