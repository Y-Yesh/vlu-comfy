import logging
import os
import io
import json
import tempfile
import uuid
import base64
import time
import requests
from typing import Tuple, Dict, Any, Union, Optional
from PIL import Image
import torch
import numpy as np


def comfyui_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI IMAGE format back to PIL Image.
    
    Args:
        image_tensor: Image tensor in ComfyUI format (batch, height, width, channels)
    
    Returns:
        PIL.Image: RGB image
    """
    # Remove batch dimension and convert to numpy
    # Shape: (1, height, width, channels) -> (height, width, channels)
    image_array = image_tensor.squeeze(0).numpy()
    
    # Denormalize from [0, 1] to [0, 255]
    image_array = (image_array * 255.0).astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(image_array)

def comfyui_image_to_binary(
    image_tensor: torch.Tensor, 
    format: str = 'PNG'
) -> bytes:
    """
    Convert ComfyUI IMAGE format to binary data.
    
    Args:
        image_tensor: Image tensor in ComfyUI format (batch, height, width, channels)
        format: Output format ('PNG', 'JPEG', etc.)
    
    Returns:
        bytes: Binary image data
    """
    pil_image = comfyui_image_to_pil(image_tensor)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    return buffer.getvalue()

def binary_to_comfyui_image(
    binary_data: bytes, 
    target_size: Tuple[int, int] = None
) -> torch.Tensor:
    """
    Convert binary image data to ComfyUI IMAGE format.
    
    Args:
        binary_data: Raw binary data of an image (JPEG, PNG, etc.)
        target_size: Optional (width, height) to resize the image. If None, keeps original size.
    
    Returns:
        torch.Tensor: Image in ComfyUI format (batch, height, width, channels)
                     - Shape: (1, height, width, 3)
                     - Data type: torch.float32
                     - Value range: [0.0, 1.0]
                     - Channels: RGB
    """
    # Load image from binary data
    image = Image.open(io.BytesIO(binary_data))
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if target size is specified
    """if target_size is not None:
        width, height = target_size
        image = image.resize((width, height), Image.Resampling.LANCZOS)"""
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to torch tensor and add batch dimension
    # Shape: (height, width, channels) -> (1, height, width, channels)
    image_tensor = torch.from_numpy(image_array)[None, :]
    
    return image_tensor

def _save_temp(filename: str, data: bytes) -> str:
    """Save bytes to a temp file and return its path."""
    _, ext = os.path.splitext(filename or "")
    if not ext:
        ext = ".bin"
    path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path

def _to_png_bytes(data: bytes) -> Tuple[bytes, str]:
    """
    Convert arbitrary image bytes to PNG bytes using Pillow.
    Returns (png_bytes, suggested_filename).
    """
    if Image is None:
        # Pillow not available; return original (caller will still label as PNG).
        # Prefer installing Pillow to guarantee correct PNG output.
        return data, "result.png"

    with Image.open(io.BytesIO(data)) as im:
        # Convert mode if needed (e.g., preserve alpha when present)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            converted = im.convert("RGBA")
        else:
            converted = im.convert("RGB")
        out = io.BytesIO()
        converted.save(out, format="PNG", optimize=True)
        return out.getvalue(), "result.png"
    

def encode_image(image_input):
    """
    Encode image to base64. Can accept either:
    - image_path (str): Path to image file
    - image_data (bytes): Binary image data
    """
    if isinstance(image_input, str):
        # Handle file path
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_input, bytes):
        # Handle binary data directly
        return base64.b64encode(image_input).decode("utf-8")
    else:
        raise ValueError("image_input must be either a file path (str) or binary data (bytes)")

def common_upscale(samples, width, height, upscale_method, crop):
        orig_shape = tuple(samples.shape)
        if len(orig_shape) > 4:
            samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
            samples = samples.movedim(2, 1)
            samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
        if crop == "center":
            old_width = samples.shape[-1]
            old_height = samples.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        else:
            s = samples

        if upscale_method == "bislerp":
            out = bislerp(s, width, height)
        elif upscale_method == "lanczos":
            out = lanczos(s, width, height)
        else:
            out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        if len(orig_shape) == 4:
            return out

        out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
        return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))
    
def get_aspect_ratio(image):
    """
    Get the aspect ratio of an image.
    Returns a tuple of (width, height, aspect_ratio, aspect_ratio_string)
    """
    if len(image.shape) == 4:  # Batch dimension present
        height, width = image.shape[1], image.shape[2]
    else:  # No batch dimension
        height, width = image.shape[0], image.shape[1]
    
    aspect_ratio = width / height
    
    # Convert to common aspect ratio string
    if abs(aspect_ratio - 1.0) < 0.01:
        aspect_ratio_str = "1:1"
    elif abs(aspect_ratio - 16/9) < 0.01:
        aspect_ratio_str = "16:9"
    elif abs(aspect_ratio - 9/16) < 0.01:
        aspect_ratio_str = "9:16"
    elif abs(aspect_ratio - 4/3) < 0.01:
        aspect_ratio_str = "4:3"
    elif abs(aspect_ratio - 3/4) < 0.01:
        aspect_ratio_str = "3:4"
    elif abs(aspect_ratio - 3/2) < 0.01:
        aspect_ratio_str = "3:2"
    elif abs(aspect_ratio - 2/3) < 0.01:
        aspect_ratio_str = "2:3"
    else:
        # Find the closest common ratio
        common_ratios = {
            "1:1": 1.0,
            "16:9": 16/9,
            "9:16": 9/16,
            "4:3": 4/3,
            "3:4": 3/4,
            "3:2": 3/2,
            "2:3": 2/3,
            "21:9": 21/9,
            "9:21": 9/21
        }
        
        closest_ratio = min(common_ratios.items(), key=lambda x: abs(x[1] - aspect_ratio))
        if abs(closest_ratio[1] - aspect_ratio) < 0.1:  # Within 10% tolerance
            aspect_ratio_str = closest_ratio[0]
        else:
            aspect_ratio_str = f"{width}:{height}"
    
    return width, height, aspect_ratio, aspect_ratio_str

def scale(image):
    width = image.shape[2]
    height = image.shape[1]
    aspect_ratio = width / height
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
    image = common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
    return image


def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        #norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        #normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        #zero when norms are zero
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0

        #slerp
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        #technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)

        #edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)

    #linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    #linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)
    
def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)
    
PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

CHARACTER_PROMPT = """ 

You are a FLUX Kontext prompt optimization system specialized in adding characters to rooms and outdoor spaces with proper dimensional scaling. Follow these rules:

CORE STRUCTURE:
"Add [character description] to [specific location in scene], positioning them [spatial relationship]. Preserve the original lighting, perspective, and environmental details of the scene."

DIMENSIONAL ANALYSIS REQUIREMENTS:
- Always identify at least 2 scale anchors in the scene (furniture, doorways, trees, buildings, objects)
- Reference these anchors explicitly for character sizing
- Specify the character's position relative to these anchors
- Include depth and perspective cues

CHARACTER DESCRIPTION TEMPLATE:
- Use specific physical descriptors: "[age/gender] with [hair color/style], wearing [clothing description]"
- Include pose/action: "standing," "sitting," "walking," or specific gesture
- Never use pronouns or vague references


Outfit Description:

Clothing Style: All characters must be dressed in expensive-looking linen vacation clothes and men in always casual loafers and women in heels or flats


SPATIAL POSITIONING FRAMEWORK:
- "in front of [anchor]" / "behind [anchor]" / "next to [anchor]"



PRESERVATION REQUIREMENTS:
- "while preserving the original scene composition"
- "maintaining the existing lighting and shadows"
- "keeping all environmental elements unchanged"
- "preserving the camera angle and perspective"

EXAMPLE OUTPUT FORMAT:
"Add a tall man with brown hair wearing a blue shirt to the living room, positioning him standing next to the coffee table. Preserve the original lighting, room layout, and camera perspective while ensuring natural integration into the scene."

QUALITY CONTROLS:
- Maximum 512 tokens per prompt
- Include at least 2 specific scale references
- Use present tense action verbs
- Specify exact positioning
- Always include preservation phrases
- Be explicit about lighting integration


"""

CHARACTER_PROMPT_MASKED = """ 

You are an expert prompt generator for masked image editing tasks. For each image provided, always follow these rules:

1. **Scene and mask analysis:**
    - Describe the overall environment of the image (lighting, furniture, key colors, visible objects, perspective).
    - Clearly note where the mask area is placed (left/right/center/foreground/background) and what it overlaps.
2. **User instruction parsing:**
    - Accept user specifications for person/object to place in the masked region: gender, age, ethnicity, clothing style, posture (standing/sitting), and any extra detail (expression, vibe, pose).
    - Parse these inputs without subheadings—use only concise, highly specific sentences.
3. **Prompt generation:**
    - Write a context-rich block similar to "[Kontext]" describing the scene and mask (max 4–5 sentences).
    - Write an "[Instruction]" block that follows the user’s requirements and always explicitly asks for natural proportions, realistic blending, and seamless integration into the environment.
4. **Realism guarantee:**
    - Make sure every instruction contains a line: “Keep the size, proportion, and lighting of the inserted subject perfectly natural for the scene and mask area.”

**Output format:**

For every image, always reply only with the following strict format and nothing extra:

`text`

`[Kontext]
(Scene analysis)

[Instruction]
(User instruction, blending/realism guidance)`

- * Sample input to the system: **
- Image with a mask covering the center of a hotel room bed
- User wants: "Mexican woman, late 20s, sitting"
- * Sample output: **

`text`

`[Kontext]
Photo of a modern hotel room with two beds, turquoise throws, sunlight, and a pink mask area on the left bed.

[Instruction]
Replace the pink masked area with a realistic Mexican woman in her late 20s, sitting comfortably on the left bed. Keep the size, proportion, and lighting perfectly natural for the environment.`

No explanations, no extra comments—just output like above every time


"""



Color_Grade_Kontext_Prompt = """

Re-light any architectural or interior image (hotel rooms, lounges, façades, poolside, balconies) to achieve a neutral-subtle cool, cinematic 3-4 late afternoon ambience inspired by luxury resort imagery.
The objective is clarity, vibrancy, and editorial-level brightness — not darkness, not muted, and absolutely no late-evening or dusky tones. The lighting must enhance realism through directional shadows and highlights but without flattening or dulling the scene. Everything should look vibrant.
If characters or humans are present, apply the re-lighting naturally to their skin, hair, and clothing while preserving their exact identity, facial features, expressions, poses, clothing details, and any accessories—do not change or alter anything about them beyond the lighting adjustment.

🔑 CORE INSTRUCTION (Copy-Ready)
Re-light the scene to create a natural late afternoon atmosphere with soft,entering from <DIRECTION>.

Cast realistic, elongated but soft shadows aligned with this light direction.

Keep overall colour temperature very subtle cool-neutral — soft daylight turquoise and pale neutrals, with no warm, yellow, or orange tones dark tones.

If the original image appears dark, dull, or underexposed, increase brightness to achieve editorial clarity and cinematic vibrancy.

Retain original material colours, textures, architectural details, and furniture precisely.
DO NOT CHANGE THE TEXTURES OF ANYTHING, DO NOT CHANGE THE TEXTURE DARK SPOTS OR ANYTHING OF THE FLOORING 
If characters or humans are present, ensure the lighting casts natural soft shadows and highlights on their skin, hair, and clothing without altering their appearance, identity, expressions, poses, or any other details—preserve everything exactly as in the original.

Preserve the existing camera angle, framing, depth of field, and spatial composition.

Skies (if visible) should reflect soft blue-gray hues with clean daylight — no burnouts, no post-sunset tones.

Water (if present) should reflect the sky naturally, maintaining soft cool tones.
The goal is clean, clear, fresh, and cinematic — matching premium hospitality references, with zero darkness or artificial grading.

🎯 STRUCTURE FOR CUSTOM USE
Change the lighting to a natural late afternoon atmosphere [add direction, e.g., from left, from right].
Ensure realistic shadows follow this direction. Maintain a cool-neutral palette with soft turquoise-gray undertones.
Increase brightness if necessary for clear, vibrant, editorial-level clarity. Preserve original colours, materials, textures, and layout.
If characters or humans appear, apply directional lighting naturally to them (e.g., soft highlights on faces and subtle shadows on bodies) while keeping their exact facial structure, skin tone, clothing, poses, and expressions unchanged.

✅ MUST-CHECK CRITERIA BEFORE FINAL OUTPUT
Lighting Must Achieve	Criteria
Light Direction	Low-angle (≈15°–25° above horizon), entering realistically
Colour Temperature	Neutral-cool, soft turquoise/grays, no warmth
Shadows	Elongated, soft-edged, matching new light direction
Overall Scene Brightness	Clean, clear, vibrant — no dull, dark, or moody patches
Sky (if visible)	Pale blue-gray, clear daylight (never dark, burnt out, or warm)
Water (if visible)	Reflecting cool daylight naturally, no harsh edits
Materials / Textures / Colours	Fully preserved, no unwanted shifts
Characters / Humans (if present)	Natural lighting applied without changing identity, features, clothing, poses, or expressions
⚠️ TROUBLESHOOT QUICK FIXES
Problem	Add This to Prompt
Scene looks too warm	“… while keeping all tones cool-neutral and avoiding warm/yellow/orange hues.”
Image is underexposed or dull	“… increase brightness to achieve a vibrant, clean, editorial clarity.”
Light leaks through walls	“… respect existing architecture; do not place light through solid walls.”
Shadows missing or unrealistic	“… ensure directional soft shadows aligned naturally with the light source.”
Characters look altered	“… preserve exact human features, skin tones, clothing, and poses without any changes beyond natural lighting.”
✍️ EXAMPLE PROMPTS
Interior, windows left:
“Re-light the room to achieve a natural late afternoon atmosphere with soft sunlight entering from the left. Let the light cast elongated shadows across the neutral flooring and furniture while preserving all existing colours and materials. Keep the overall palette cool-neutral with turquoise-gray undertones. Increase brightness if necessary to achieve a clean, cinematic, editorial clarity. If any characters are present, apply the lighting naturally to their skin and clothing without changing their appearance, expressions, or poses.”

Exterior, poolside terrace:
“Shift the lighting to a natural late afternoon scene with soft sunlight from the horizon behind the pool. Ensure shadows stretch across the terrace and water, keeping colours cool-neutral with no warmth. Preserve all textures and materials. Maintain vibrant, editorial brightness with soft turquoise-gray skies reflected in the water. For any humans in the scene, cast realistic highlights and shadows on them while keeping their exact identity, clothing, and positions unchanged.”

🚨 ABSOLUTE RULES
NO warm overtones (yellow-orange cast is prohibited)

NO underexposure (dull or dark is not premium)

NO alterations to materials, objects, or angles

NO adding “evening mood” filters — natural light realism only

NO visual flattening through over-smoothing or haze

NO changes to characters/humans (preserve identity, features, clothing, poses exactly—only apply natural lighting)

🎛 VERB LIST FOR VARIATION
Use flexibly depending on tone:

Change

Shift

Adjust

Re-light

Transform

🔍 YOUR SYSTEM TASK (AUTOMATED EXPECTATION):
You must analyze the image first.
If the base image is already dark, dull, or lacking clarity:
➡️ Automatically adjust the prompt to prioritize vibrancy, clarity, and editorial brightness.
➡️ Add realistic soft shadows where necessary but never allow the image to look darker or moodier than luxury hospitality standards require.
➡️ Prioritize a clean, cinematic, editorial finish — never “evening mood,” never “sunset,” never muted.
If characters or humans are detected, ensure prompts include preservation clauses to maintain their exact appearance while integrating natural lighting effects seamlessly.

🚫 COMMON MISTAKES TO AVOID
Avoid: “Golden-hour warmth” overlays — this is a cool palette, not warm.

Avoid: Over-softening or hazing the scene in the name of ‘cinema’ — clarity > mood.

Avoid: Generic sunset vibes — this is 4–5 PM clean, not dusk.

Avoid: Unintended changes to humans — always specify “preserve exact features and clothing” when characters are present.

Example prompt for reference: Re-light this bedroom scene to achieve a clean, crisp, late afternoon (3–4 PM) effect with soft, low-angle light entering from the balcony.
Create elongated soft shadows stretching across the stone floor from left to right.
Lift the midtones on the bed and floor slightly for clarity.
Ensure all glass and water reflections align with cool-neutral daylight (turquoise-gray, no warmth).
Brighten overall scene slightly for editorial clarity without altering textures.
Maintain soft sky with clear blue-gray hues.
If any characters are in the scene, apply the lighting naturally to their faces and bodies while preserving their exact expressions, clothing, and poses unchanged.
The final image should feel fresh, cinematic, and premium — as seen in luxury hospitality visuals.Re-light any architectural or interior image (hotel rooms, lounges, façades, poolside, balconies) to achieve a user-specified time-of-day lighting effect — including morning, afternoon, golden hour, sunset, overcast, or night. All outputs must reflect realistic, high-end hospitality standards: clear, vibrant, architectural, and cinematic — never stylized, never fantasy.

🔑 CORE INSTRUCTION (Copy-Ready)
Re-light the scene to reflect a [TIME OF DAY / LIGHT TYPE] atmosphere with natural light entering from [DIRECTION], or appropriate artificial lighting for night scenes.
• Cast realistic shadows aligned with the light source or create balanced ambient lighting for night.
• Adapt colour temperature to match the chosen light (see table below).
• Increase brightness where necessary to maintain clarity and editorial-level vibrancy.
• Preserve original materials, textures, architectural details, and layout.
• Maintain existing camera angle, framing, depth of field, and spatial logic.
• Skies (if visible) must match realistic tones for the time of day — never artificial gradients.
• Water (if visible) must reflect sky or lighting naturally.
• At night: Use architectural lighting sources (lamps, sconces, pool lights) — clean, neutral-white, soft ambient light only.
• Overall objective: premium, clean, cinematic clarity aligned with global luxury hospitality standards.

🎨 LIGHTING TEMPERATURE GUIDE (Reference)
Time of Day / Light Type	Temperature Tone	Mood
Morning (7–9 AM)	Neutral-cool (pale blue-gray)	Fresh, bright
Midday (11–1 PM)	Neutral (white daylight)	Bright, clear
Afternoon (3–4 PM)	Neutral-cool (blue-gray)	Crisp, soft
Golden Hour (5–6 PM)	Warm-neutral (soft amber-peach)	Gentle warmth
Overcast / Cloudy	Cool-neutral (blue-gray)	Diffused, soft
Night (9 PM onwards)	Neutral-warm (soft architectural white light)	Calm, ambient, clean

✅ MUST-CHECK CRITERIA BEFORE FINAL OUTPUT
Lighting Must Achieve	Criteria
Light Direction	Aligned with user input or logical architectural sources at night
Colour Temperature	Matches light type: cool / neutral / warm / artificial night light
Shadows	Realistic, soft-edged where appropriate, ambient at night
Scene Brightness	Clean, clear, editorial brightness (no darkness or haze)
Sky (if visible)	Natural, correct for time of day; deep blue at night, no gradients
Water (if visible)	Natural reflection; at night, soft highlights from architectural light
Materials / Textures / Colours	Fully preserved, no unintended shifts

⚠ TROUBLESHOOT QUICK FIXES
Problem	Solution to Add to Prompt
Scene too warm / too cool	“… match the specified light tone precisely; avoid unintended warmth or coolness.”
Underexposed or dull image	“… increase brightness to achieve clear, editorial clarity without flattening.”
Light leaks through walls	“… respect existing architecture; light must not pass through solid structures.”
Missing or unrealistic shadows	“… ensure natural, directional soft shadows aligned with light source.”
Night lighting feels unrealistic	“… use clean architectural light sources only; avoid harsh or stylized effects.”

✍ EXAMPLE PROMPTS (Including Night)
Morning Light, Left Window:
Re-light the room for a fresh morning (7–9 AM) effect with soft natural light entering from the left window. Cast soft shadows on the floor and furniture. Maintain a cool-neutral palette with blue-gray undertones. Increase brightness for clear editorial clarity. Sky outside should reflect pale blue-gray morning tones.

Golden Hour, Right Side:
Re-light the scene for a gentle golden hour (5–6 PM) effect with warm sunlight entering from the right. Cast elongated, soft shadows. Balance tones with amber warmth while preserving material colours. Increase brightness if needed for cinematic clarity. Sky (if visible) should reflect soft peach gradients.

Overcast Noon, Front-Facing:
Adjust for a cloudy midday with diffused light entering from the front-facing windows. Soft, minimal shadows. Brighten for editorial clarity. Cool-neutral palette. Sky pale blue-gray.

Night Scene, Interior:
Re-light the interior for a clean architectural night scene with soft ambient light from overhead fixtures and lamps. Ensure natural pools of light on floors and surfaces. No harsh shadows, no warm-orange tones — only neutral-white, soft hospitality-standard lighting. Windows should show deep blue night sky, no gradients.

Night Scene, Exterior Poolside:
Re-light the terrace and pool for a nighttime resort look with soft architectural lights from sconces and pool lights. Cast realistic glows and soft shadows. Water should reflect cool lighting gently. Maintain clarity and texture in all surfaces; no harsh spotlights.

🚨 ABSOLUTE RULES
NO warmth unless explicitly specified (e.g., golden hour)

NO underexposure or “moody” darkness

NO altering materials, architecture, or layout

NO evening filters unless explicitly requested

NO unrealistic lens flares or gradient overlays

NO stylized artificial “cinema light” tricks — only realism

🎛 VERB LIST FOR VARIATION
Change

Shift

Adjust

Re-light

Transform

🔍 SYSTEM TASK (AUTOMATED EXPECTATION):
Analyze user input and base image. Adapt output precisely for:

Specified time of day / light type

Correct shadows and light sources (natural or architectural)

Correct colour temperature

Clarity, vibrancy, editorial brightness

🚫 COMMON MISTAKES TO AVOID
Avoid yellow/orange warmth unless golden hour requested

Avoid haze or softening that removes clarity

Avoid pink/magenta gradients unless requested

Avoid night scenes that feel artificially dark or gloomy
THE IMAGE SHOULD NEVER BE TOO SATURATED

"""
    
def poll_for_result(polling_url: str, headers: Dict[str, str], max_attempts: int = 30, delay: float = 2.0) -> Optional[bytes]:
    """
    Poll the Flux API polling URL until the image generation is complete.
    
    Args:
        polling_url: The URL to poll for results
        headers: Headers to use for the request
        max_attempts: Maximum number of polling attempts
        delay: Delay between polling attempts in seconds
    
    Returns:
        The generated image as bytes, or None if polling fails
    """
    for attempt in range(max_attempts):
        try:
            logging.info(f"Polling attempt {attempt + 1}/{max_attempts}")
            response = requests.get(polling_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logging.info(f"Polling response: {data}")
                
                # Check if the generation is complete
                if data.get("status") in ["ready", "Ready"]:
                    # Get the image URL from the response
                    if data.get("result") and "sample" in data["result"]:
                        image_url = data["result"]["sample"]
                        logging.info(f"Image ready at: {image_url}")
                        
                        # Download the image
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            return img_response.content
                        else:
                            logging.error(f"Failed to download image: {img_response.status_code}")
                            return None
                    else:
                        logging.error("No 'sample' field in completed response")
                        return None
                        
                elif data.get("status") in ["failed", "Failed"]:
                    logging.error(f"Image generation failed: {data}")
                    return None
                    
                elif data.get("status") in ["pending", "Pending", "processing", "Processing"]:
                    logging.info(f"Still processing... status: {data.get('status')}")
                    time.sleep(delay)
                    continue
                else:
                    logging.warning(f"Unknown status: {data.get('status')}")
                    time.sleep(delay)
                    continue
                    
            else:
                logging.error(f"Polling failed with status code: {response.status_code}")
                time.sleep(delay)
                continue
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error during polling: {e}")
            time.sleep(delay)
            continue
        except Exception as e:
            logging.error(f"Unexpected error during polling: {e}")
            time.sleep(delay)
            continue
    
    logging.error(f"Polling timed out after {max_attempts} attempts")
    return None

