import azure.functions as func
import logging
import os
import io
import json
import tempfile
import uuid
import base64
import random
from typing import Tuple
from PIL import Image
from openai import OpenAI
import requests


from helps import _save_temp, _to_png_bytes, encode_image, common_upscale, PREFERED_KONTEXT_RESOLUTIONS, Angle_Kontext_Prompt, Color_Grade_Kontext_Prompt, binary_to_comfyui_image, scale, comfyui_image_to_binary, get_aspect_ratio, poll_for_result

# Maximum file size in bytes (10 MB)
MAX_BYTES = 10 * 1024 * 1024

# Camera angle options for AI angle change
CAMERA_ANGLE_OPTIONS = [
    "Slight Pan left",
    "Slight Pan right",
    "Tilt up",
    "Tilt down",
    "Eye-level shot",
    "Top-down"
]

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
client = OpenAI(api_key="")
url = "https://api.bfl.ai/v1/flux-kontext-max"

payload = {
    "prompt": "ein fantastisches bild",
    "input_image": "<string>",
    "seed": 42,
    "aspect_ratio": "<string>",
    "output_format": "png",
    "prompt_upsampling": False,
    "safety_tolerance": 2
}

api_headers = {
    "x-key": "",
    "Content-Type": "application/json"
}

@app.route(route="angle-generate")
def create(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Start overall timing
        import time
        overall_start_time = time.time()
        # Expect multipart/form-data with fields: prompt (text) + image (file)
        files = req.files or {}
        form = req.form or {}
        
        logging.info(f"Entered")
        logging.info(f"Request: {req}")
        logging.info(f"Form: {form}")
        logging.info(f"Files: {files}")
        
        prompt = form.get("prompt", "")
        ai_angle_change = form.get("ai_angle_change", "false").lower() == "true"
        
        # If ai_angle_change is True, randomly select a camera angle
        if ai_angle_change:
            selected_angle = random.choice(CAMERA_ANGLE_OPTIONS)
            prompt = selected_angle
            logging.info(f"AI angle change enabled - randomly selected angle: {selected_angle}")
        
        logging.info(f"Prompt: {prompt}")
        #if not prompt:
            #return func.HttpResponse("Missing 'prompt' field.")

        # Check for either 'image' or 'file' field
        file_key = None
        if "image" in files:
            file_key = "image"
        elif "file" in files:
            file_key = "file"
        else:
            return func.HttpResponse("Missing 'image' or 'file' field.")

        file = files[file_key]  # FileStorage-like object
        filename = getattr(file, "filename", None) or "upload"
        data = file.stream.read()

        if not data:
            return func.HttpResponse("Empty file uploaded.")
        if len(data) > MAX_BYTES:
            return func.HttpResponse(f"File '{filename}' exceeds {MAX_BYTES//1024//1024} MB limit.", 413)

        # Get image type from FileStorage object
        file_mimetype = getattr(file, 'content_type', 'unknown')
        file_extension = os.path.splitext(filename)[1].lower() if filename else ''
        
        image_info = {
            'filename': filename,
            'mimetype': file_mimetype,
            'extension': file_extension,
            'size_bytes': len(data)
        }
        logging.info(f"File info: {image_info}")

        # (Optional) keep a temp copy if you need later (e.g., logging / retries)
        
        # Getting the Base64 string
        #base64_image = encode_image(path_temp)
        
        # Start timing for ComfyUI processing steps
        
        # Step 1: Convert binary to ComfyUI format
        step1_start = time.time()
        comfy_image = binary_to_comfyui_image(data) # convert the uploaded image to the format used by ComfyUI
        step1_end = time.time()
        step1_time = step1_end - step1_start
        logging.info(f"Step 1 - binary_to_comfyui_image: {step1_time:.3f} seconds")
        
        # Step 2: Scale the image
        step2_start = time.time()
        scaled_image = scale(comfy_image)
        step2_end = time.time()
        step2_time = step2_end - step2_start
        logging.info(f"Step 2 - scale: {step2_time:.3f} seconds")
        
        # Step 3: Convert back to binary
        step3_start = time.time()
        byte_converted_image = comfyui_image_to_binary(scaled_image)
        step3_end = time.time()
        step3_time = step3_end - step3_start
        logging.info(f"Step 3 - comfyui_image_to_binary: {step3_time:.3f} seconds")
        
        # Total ComfyUI processing time
        total_comfyui_time = step1_time + step2_time + step3_time
        logging.info(f"Total ComfyUI processing: {total_comfyui_time:.3f} seconds")

        # Get aspect ratio of scaled image
        width, height, aspect_ratio, aspect_ratio_str = get_aspect_ratio(scaled_image)
        logging.info(f"Scaled image dimensions: {width}x{height}, aspect ratio: {aspect_ratio_str} ({aspect_ratio:.3f})")
        payload["aspect_ratio"] = aspect_ratio_str

        base64_image_comfyui = encode_image(byte_converted_image)
        payload["input_image"] = f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Use the rules to generate a prompt based on the user's prompt and the image. "
                                    f"If no user prompt is provided, use the image to generate a prompt. {Angle_Kontext_Prompt}"
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt if prompt else ""
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"
                        },
                    ],
                },
            ],
        )

        
        logging.info(f"Open AI Response: {response.output_text}")
        payload["prompt"] = response.output_text
        # Here you could call Freepik with (data, prompt) and get edited bytes back.
        # For now, we just convert the uploaded image to PNG and return it inline.
        response = requests.post(url, json=payload, headers=api_headers)
        response_data = response.json()
        logging.info(f"Flux Response: {response_data}")
        
        if "polling_url" in response_data:
            polling_url = response_data["polling_url"]
            logging.info(f"Polling URL: {polling_url}")
            
            # Poll the URL until the image is ready
            result_image = poll_for_result(polling_url, api_headers)
            
            if result_image:
                # Convert the result image to bytes for response
                logging.info(f"Result image Sent to the frontend")
                png_bytes, out_name = _to_png_bytes(result_image)
            else:
                # Fallback to original processed image if polling fails
                logging.error("No result image in response")
                png_bytes, out_name = _to_png_bytes(byte_converted_image)
        else:
            logging.error("No polling_url in response")
            # Fallback to original processed image
            png_bytes, out_name = _to_png_bytes(byte_converted_image)
        
        # Inline PNG response for immediate preview
        response_headers = {
            "Content-Disposition": f'inline; filename="{out_name}"',
            # Add CORS if your frontend is on another origin:
            # "Access-Control-Allow-Origin": "*",
        }
        
        # Calculate total function execution time
        overall_end_time = time.time()
        total_function_time = overall_end_time - overall_start_time
        logging.info(f"Total function execution time: {total_function_time:.3f} seconds")
        
        return func.HttpResponse(
            body=png_bytes,
            status_code=200,
            mimetype="image/png",
            headers=response_headers
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="angle-generate-batch")
def create_batch(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handle folder uploads with multiple images.
    Expects multipart/form-data with:
    - prompt (optional text)
    - files (multiple image files)
    """
    try:
        import time
        overall_start_time = time.time()
        
        # Get form data and files
        files = req.files or {}
        form = req.form or {}
        
        logging.info(f"Batch upload entered")
        logging.info(f"Form: {form}")
        logging.info(f"Files: {list(files.keys())}")
        
        prompt = form.get("prompt", "")
        ai_angle_change = form.get("ai_angle_change", "false").lower() == "true"
        
        # If ai_angle_change is True, randomly select a camera angle
        if ai_angle_change:
            selected_angle = random.choice(CAMERA_ANGLE_OPTIONS)
            prompt = selected_angle
            logging.info(f"AI angle change enabled - randomly selected angle: {selected_angle}")
        
        logging.info(f"Prompt: {prompt}")
        
        # Collect all image files
        image_files = []
        for key, file in files.items():
            if key.startswith(('image', 'file')) or key in ['image', 'file']:
                filename = getattr(file, "filename", None) or f"upload_{key}"
                data = file.stream.read()
                
                if data and len(data) <= MAX_BYTES:
                    file_mimetype = getattr(file, 'content_type', 'unknown')
                    file_extension = os.path.splitext(filename)[1].lower() if filename else ''
                    
                    image_info = {
                        'filename': filename,
                        'mimetype': file_mimetype,
                        'extension': file_extension,
                        'size_bytes': len(data),
                        'data': data
                    }
                    image_files.append(image_info)
                    logging.info(f"Added file: {filename} ({len(data)} bytes)")
                else:
                    if len(data) > MAX_BYTES:
                        logging.warning(f"File {filename} exceeds size limit, skipping")
                    else:
                        logging.warning(f"Empty file {filename}, skipping")
        
        if not image_files:
            return func.HttpResponse(
                json.dumps({"error": "No valid image files found"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logging.info(f"Processing {len(image_files)} images")
        
        # Process each image
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, image_info in enumerate(image_files):
            try:
                logging.info(f"Processing image {i+1}/{len(image_files)}: {image_info['filename']}")
                
                # Step 1: Convert binary to ComfyUI format
                step1_start = time.time()
                comfy_image = binary_to_comfyui_image(image_info['data'])
                step1_end = time.time()
                step1_time = step1_end - step1_start
                
                # Step 2: Scale the image
                step2_start = time.time()
                scaled_image = scale(comfy_image)
                step2_end = time.time()
                step2_time = step2_end - step2_start
                
                # Step 3: Convert back to binary
                step3_start = time.time()
                byte_converted_image = comfyui_image_to_binary(scaled_image)
                step3_end = time.time()
                step3_time = step3_end - step3_start
                
                # Get aspect ratio
                width, height, aspect_ratio, aspect_ratio_str = get_aspect_ratio(scaled_image)
                
                # Prepare payload for this image
                image_payload = payload.copy()
                image_payload["aspect_ratio"] = aspect_ratio_str
                
                base64_image_comfyui = encode_image(byte_converted_image)
                image_payload["input_image"] = f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"
                
                # Generate prompt using OpenAI
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"Use the rules to generate a prompt based on the user's prompt and the image. "
                                            f"If no user prompt is provided, use the image to generate a prompt. {Angle_Kontext_Prompt}"
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt if prompt else ""
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"
                                },
                            ],
                        },
                    ],
                )
                
                image_payload["prompt"] = response.output_text
                
                # Call Flux API
                flux_response = requests.post(url, json=image_payload, headers=api_headers)
                flux_response_data = flux_response.json()
                
                if "polling_url" in flux_response_data:
                    polling_url = flux_response_data["polling_url"]
                    result_image = poll_for_result(polling_url, api_headers)
                    
                    if result_image:
                        png_bytes, out_name = _to_png_bytes(result_image)
                        results.append({
                            "filename": image_info['filename'],
                            "processed_filename": out_name,
                            "image_data": base64.b64encode(png_bytes).decode('utf-8'),
                            "status": "success",
                            "processing_time": step1_time + step2_time + step3_time
                        })
                        successful_count += 1
                        logging.info(f"Successfully processed: {image_info['filename']}")
                    else:
                        results.append({
                            "filename": image_info['filename'],
                            "status": "failed",
                            "error": "Failed to get result from Flux API"
                        })
                        failed_count += 1
                        logging.error(f"Failed to get result for: {image_info['filename']}")
                else:
                    results.append({
                        "filename": image_info['filename'],
                        "status": "failed",
                        "error": "No polling URL in Flux response"
                    })
                    failed_count += 1
                    logging.error(f"No polling URL for: {image_info['filename']}")
                    
            except Exception as e:
                results.append({
                    "filename": image_info['filename'],
                    "status": "failed",
                    "error": str(e)
                })
                failed_count += 1
                logging.error(f"Error processing {image_info['filename']}: {str(e)}")
        
        # Calculate total execution time
        overall_end_time = time.time()
        total_function_time = overall_end_time - overall_start_time
        
        # Prepare response
        response_data = {
            "total_images": len(image_files),
            "successful": successful_count,
            "failed": failed_count,
            "total_processing_time": total_function_time,
            "results": results
        }
        
        logging.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")
        logging.info(f"Total batch processing time: {total_function_time:.3f} seconds")
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Batch processing error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="color-grade-generate")
def create_color_grade(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Start overall timing
        import time
        overall_start_time = time.time()
        # Expect multipart/form-data with fields: prompt (text) + image (file)
        files = req.files or {}
        form = req.form or {}
        
        logging.info(f"Entered")
        logging.info(f"Request: {req}")
        logging.info(f"Form: {form}")
        logging.info(f"Files: {files}")
        
        prompt = form.get("prompt", "")
        
        
        logging.info(f"Prompt: {prompt}")
        #if not prompt:
            #return func.HttpResponse("Missing 'prompt' field.")

        # Check for either 'image' or 'file' field
        file_key = None
        if "image" in files:
            file_key = "image"
        elif "file" in files:
            file_key = "file"
        else:
            return func.HttpResponse("Missing 'image' or 'file' field.")

        file = files[file_key]  # FileStorage-like object
        filename = getattr(file, "filename", None) or "upload"
        data = file.stream.read()

        if not data:
            return func.HttpResponse("Empty file uploaded.")
        if len(data) > MAX_BYTES:
            return func.HttpResponse(f"File '{filename}' exceeds {MAX_BYTES//1024//1024} MB limit.", 413)

        # Get image type from FileStorage object
        file_mimetype = getattr(file, 'content_type', 'unknown')
        file_extension = os.path.splitext(filename)[1].lower() if filename else ''
        
        image_info = {
            'filename': filename,
            'mimetype': file_mimetype,
            'extension': file_extension,
            'size_bytes': len(data)
        }
        logging.info(f"File info: {image_info}")

        # (Optional) keep a temp copy if you need later (e.g., logging / retries)
        
        # Getting the Base64 string
        #base64_image = encode_image(path_temp)
        
        # Start timing for ComfyUI processing steps
        
        # Step 1: Convert binary to ComfyUI format
        step1_start = time.time()
        comfy_image = binary_to_comfyui_image(data) # convert the uploaded image to the format used by ComfyUI
        step1_end = time.time()
        step1_time = step1_end - step1_start
        logging.info(f"Step 1 - binary_to_comfyui_image: {step1_time:.3f} seconds")
        
        # Step 2: Scale the image
        step2_start = time.time()
        scaled_image = scale(comfy_image)
        step2_end = time.time()
        step2_time = step2_end - step2_start
        logging.info(f"Step 2 - scale: {step2_time:.3f} seconds")
        
        # Step 3: Convert back to binary
        step3_start = time.time()
        byte_converted_image = comfyui_image_to_binary(scaled_image)
        step3_end = time.time()
        step3_time = step3_end - step3_start
        logging.info(f"Step 3 - comfyui_image_to_binary: {step3_time:.3f} seconds")
        
        # Total ComfyUI processing time
        total_comfyui_time = step1_time + step2_time + step3_time
        logging.info(f"Total ComfyUI processing: {total_comfyui_time:.3f} seconds")

        # Get aspect ratio of scaled image
        width, height, aspect_ratio, aspect_ratio_str = get_aspect_ratio(scaled_image)
        logging.info(f"Scaled image dimensions: {width}x{height}, aspect ratio: {aspect_ratio_str} ({aspect_ratio:.3f})")
        payload["aspect_ratio"] = aspect_ratio_str

        base64_image_comfyui = encode_image(byte_converted_image)
        payload["input_image"] = f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Use the rules to generate a prompt based on the user's prompt and the image. "
                                    f"{Color_Grade_Kontext_Prompt}"
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt if prompt else ""
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"
                        },
                    ],
                },
            ],
        )

        
        logging.info(f"Open AI Response: {response.output_text}")
        payload["prompt"] = response.output_text
        # Here you could call Freepik with (data, prompt) and get edited bytes back.
        # For now, we just convert the uploaded image to PNG and return it inline.
        response = requests.post(url, json=payload, headers=api_headers)
        response_data = response.json()
        logging.info(f"Flux Response: {response_data}")
        
        if "polling_url" in response_data:
            polling_url = response_data["polling_url"]
            logging.info(f"Polling URL: {polling_url}")
            
            # Poll the URL until the image is ready
            result_image = poll_for_result(polling_url, api_headers)
            
            if result_image:
                # Convert the result image to bytes for response
                logging.info(f"Result image Sent to the frontend")
                png_bytes, out_name = _to_png_bytes(result_image)
            else:
                # Fallback to original processed image if polling fails
                logging.error("No result image in response")
                png_bytes, out_name = _to_png_bytes(byte_converted_image)
        else:
            logging.error("No polling_url in response")
            # Fallback to original processed image
            png_bytes, out_name = _to_png_bytes(byte_converted_image)
        
        # Inline PNG response for immediate preview
        response_headers = {
            "Content-Disposition": f'inline; filename="{out_name}"',
            # Add CORS if your frontend is on another origin:
            # "Access-Control-Allow-Origin": "*",
        }
        
        # Calculate total function execution time
        overall_end_time = time.time()
        total_function_time = overall_end_time - overall_start_time
        logging.info(f"Total function execution time: {total_function_time:.3f} seconds")
        
        return func.HttpResponse(
            body=png_bytes,
            status_code=200,
            mimetype="image/png",
            headers=response_headers
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="color-grade-generate-batch")
def create_color_grade_batch(req: func.HttpRequest) -> func.HttpResponse:
    """
    Handle folder uploads with multiple images.
    Expects multipart/form-data with:
    - prompt (optional text)
    - files (multiple image files)
    """
    try:
        import time
        overall_start_time = time.time()
        
        # Get form data and files
        files = req.files or {}
        form = req.form or {}
        
        logging.info(f"Batch upload entered")
        logging.info(f"Form: {form}")
        logging.info(f"Files: {list(files.keys())}")
        
        prompt = form.get("prompt", "")
        ai_angle_change = form.get("ai_angle_change", "false").lower() == "true"
        
        # If ai_angle_change is True, randomly select a camera angle
        if ai_angle_change:
            selected_angle = random.choice(CAMERA_ANGLE_OPTIONS)
            prompt = selected_angle
            logging.info(f"AI angle change enabled - randomly selected angle: {selected_angle}")
        
        logging.info(f"Prompt: {prompt}")
        
        # Collect all image files
        image_files = []
        for key, file in files.items():
            if key.startswith(('image', 'file')) or key in ['image', 'file']:
                filename = getattr(file, "filename", None) or f"upload_{key}"
                data = file.stream.read()
                
                if data and len(data) <= MAX_BYTES:
                    file_mimetype = getattr(file, 'content_type', 'unknown')
                    file_extension = os.path.splitext(filename)[1].lower() if filename else ''
                    
                    image_info = {
                        'filename': filename,
                        'mimetype': file_mimetype,
                        'extension': file_extension,
                        'size_bytes': len(data),
                        'data': data
                    }
                    image_files.append(image_info)
                    logging.info(f"Added file: {filename} ({len(data)} bytes)")
                else:
                    if len(data) > MAX_BYTES:
                        logging.warning(f"File {filename} exceeds size limit, skipping")
                    else:
                        logging.warning(f"Empty file {filename}, skipping")
        
        if not image_files:
            return func.HttpResponse(
                json.dumps({"error": "No valid image files found"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logging.info(f"Processing {len(image_files)} images")
        
        # Process each image
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, image_info in enumerate(image_files):
            try:
                logging.info(f"Processing image {i+1}/{len(image_files)}: {image_info['filename']}")
                
                # Step 1: Convert binary to ComfyUI format
                step1_start = time.time()
                comfy_image = binary_to_comfyui_image(image_info['data'])
                step1_end = time.time()
                step1_time = step1_end - step1_start
                
                # Step 2: Scale the image
                step2_start = time.time()
                scaled_image = scale(comfy_image)
                step2_end = time.time()
                step2_time = step2_end - step2_start
                
                # Step 3: Convert back to binary
                step3_start = time.time()
                byte_converted_image = comfyui_image_to_binary(scaled_image)
                step3_end = time.time()
                step3_time = step3_end - step3_start
                
                # Get aspect ratio
                width, height, aspect_ratio, aspect_ratio_str = get_aspect_ratio(scaled_image)
                
                # Prepare payload for this image
                image_payload = payload.copy()
                image_payload["aspect_ratio"] = aspect_ratio_str
                
                base64_image_comfyui = encode_image(byte_converted_image)
                image_payload["input_image"] = f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"
                
                # Generate prompt using OpenAI
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {
                            "role": "developer",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": f"Use the rules to generate a prompt based on the user's prompt and the image. "
                                            f"If no user prompt is provided, use the image to generate a prompt. {Color_Grade_Kontext_Prompt}"
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt if prompt else ""
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{image_info['mimetype']};base64,{base64_image_comfyui}"
                                },
                            ],
                        },
                    ],
                )
                
                image_payload["prompt"] = response.output_text
                
                # Call Flux API
                flux_response = requests.post(url, json=image_payload, headers=api_headers)
                flux_response_data = flux_response.json()
                
                if "polling_url" in flux_response_data:
                    polling_url = flux_response_data["polling_url"]
                    result_image = poll_for_result(polling_url, api_headers)
                    
                    if result_image:
                        png_bytes, out_name = _to_png_bytes(result_image)
                        results.append({
                            "filename": image_info['filename'],
                            "processed_filename": out_name,
                            "image_data": base64.b64encode(png_bytes).decode('utf-8'),
                            "status": "success",
                            "processing_time": step1_time + step2_time + step3_time
                        })
                        successful_count += 1
                        logging.info(f"Successfully processed: {image_info['filename']}")
                    else:
                        results.append({
                            "filename": image_info['filename'],
                            "status": "failed",
                            "error": "Failed to get result from Flux API"
                        })
                        failed_count += 1
                        logging.error(f"Failed to get result for: {image_info['filename']}")
                else:
                    results.append({
                        "filename": image_info['filename'],
                        "status": "failed",
                        "error": "No polling URL in Flux response"
                    })
                    failed_count += 1
                    logging.error(f"No polling URL for: {image_info['filename']}")
                    
            except Exception as e:
                results.append({
                    "filename": image_info['filename'],
                    "status": "failed",
                    "error": str(e)
                })
                failed_count += 1
                logging.error(f"Error processing {image_info['filename']}: {str(e)}")
        
        # Calculate total execution time
        overall_end_time = time.time()
        total_function_time = overall_end_time - overall_start_time
        
        # Prepare response
        response_data = {
            "total_images": len(image_files),
            "successful": successful_count,
            "failed": failed_count,
            "total_processing_time": total_function_time,
            "results": results
        }
        
        logging.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")
        logging.info(f"Total batch processing time: {total_function_time:.3f} seconds")
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Batch processing error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )       
        
