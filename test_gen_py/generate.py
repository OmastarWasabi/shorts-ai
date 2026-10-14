from openai import OpenAI
from pathlib import Path
import base64
import time

client = OpenAI()

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

prompt = """
Photorealistic cinematic historical photograph of the Great Emu War in Western Australia, 1932.

Three Australian soldiers wearing authentic 1930s khaki military uniforms and wide-brimmed slouch hats are positioned in a dry golden wheat field.

In the foreground, one soldier crouches behind a mounted Lewis light machine gun on a tripod, aiming toward a group of emus running across the field. Two other soldiers watch the birds with serious, focused expressions.

Three large emus are visible in the middle distance, running through the tall dry grass.

Hot Australian afternoon, dusty atmosphere, warm natural sunlight, muted earthy colors, realistic human faces, historically accurate clothing and military equipment.

Documentary war photography, cinematic composition, shallow depth of field, subtle film grain, natural textures, highly realistic details.

Vertical 2:3 composition, foreground soldiers and machine gun in sharp focus, emus slightly blurred in the background.

No text, no captions, no watermarks, no modern objects, no illustration, no cartoon, no CGI.
"""

print("Генерируем изображение...")

start = time.time()

# result = client.images.generate(
#     model="gpt-image-1-mini",
#     prompt=prompt,
#     size="1024x1536",
#     quality="low",
#     n=1,
#     output_format="png",
# )

raw_response = client.images.with_raw_response.generate(
    model="gpt-image-2.5-flare",
    prompt=prompt,
    size="1024x1536",
    quality="low",
    n=1,
    output_format="png",
)

print("\n=== RATE LIMITS ===")

for key, value in raw_response.headers.items():
    if "ratelimit" in key.lower():
        print(f"{key}: {value}")

result = raw_response.parse()

image_data = base64.b64decode(result.data[0].b64_json)

filename = OUTPUT_DIR / "02_mini4.png"
filename.write_bytes(image_data)

elapsed = time.time() - start

print(f"Готово: {filename}")
print(f"Время генерации: {elapsed:.1f} секунд")