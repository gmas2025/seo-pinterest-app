import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import pandas as pd
from openai import OpenAI
import requests
from PIL import Image
import io
from google.cloud import storage 
import time

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("\n--- Gemini API configured successfully. ---")
    except Exception as e:
        st.error(f"Error during initial Gemini API configuration: {e}. Please check your GEMINI_API_KEY.")
        print(f"\n--- ERROR: Initial Gemini API configuration failed: {e} ---")
        st.stop()
else:
    st.error("GEMINI_API_KEY not found in .env file. Please add it.")
    print("\n--- ERROR: GEMINI_API_KEY environment variable not found. ---")
    st.stop()

# Configure OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("--- OpenAI API client initialized. ---")
else:
    st.error("OPENAI_API_KEY not found in .env file. Please add it.")
    print("\n--- ERROR: OPENAI_API_KEY environment variable not found. ---")
    st.stop()


# --- Debugging Function ---
def list_gemini_models():
    st.subheader("Debugging: Available Gemini Models")
    try:
        print("--- Attempting to list Gemini models via genai.list_models()... ---")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append({
                    "Name": m.name,
                    "Description": m.description,
                    "Methods": ", ".join(m.supported_generation_methods),
                    "Version": m.version
                })
        
        print(f"--- Found {len(available_models)} models supporting generateContent. ---")

        if available_models:
            st.dataframe(pd.DataFrame(available_models))
            st.info("Look for models like 'gemini-pro' or 'models/gemini-pro' in the list. If 'gemini-pro' isn't there, try another 'generateContent' model.")
        else:
            st.warning("No models found that support 'generateContent'. This might indicate an API key, regional issue, or a problem with the API itself.")
    except Exception as e:
        print(f"\n--- ERROR caught in list_gemini_models function: {e} ---")
        st.error(f"Error listing Gemini models: {e}. This likely means your Gemini API key is invalid or your project has no access. Please check your terminal for more details.")


# Define the table structure for display
TABLE_HEADERS = ["Title", "Subtitle", "Hook", "Image Background", "Description", "Hashtags", "Alt Text", "Image_Url", "Status"]


# --- Function to generate SEO records using Gemini ---
def generate_seo_records_with_gemini(user_prompt):
    print("\n--- Entering generate_seo_records_with_gemini function ---")
    retries = 3 
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest') 
            print(f"--- Attempting to use Gemini model: {model.model_name} (Attempt {attempt + 1}/{retries}) ---")

            prompt_template = f"""
            You are an expert SEO content creator and Pinterest marketing specialist. Your task is to generate 10 unique, keyword-optimized, and highly relevant Pinterest pin content records based on the user's core topic.

            **User Topic:** {user_prompt}

            **Objective:**
            - Generate content that appeals to a target audience interested in the user topic.
            - Ensure all fields are rich with relevant keywords and directly derive from the user topic.
            - Maintain a professional and engaging tone.
            - The 'Image Background' description must be extremely detailed and evocative for an AI image generator (e.g., DALL-E 3, Midjourney), ensuring it accurately reflects the content.
            - Ensure variety across the 10 generated records while staying true to the core topic.

            **Output Format (Strict JSON Array of Objects - ensure valid JSON with no extra text):**
            ```json
            [
              {{
                "Title": "...", // Catchy, keyword-rich, max 60 chars. Directly related to topic.
                "Subtitle": "...", // Explanatory, keyword-rich, expands on title, max 120 chars. Directly related.
                "Hook": "...", // Engaging question or statement, max 80 chars. Designed to grab attention.
                "Image Background": "...", // *Crucial for AI image generation*. Highly descriptive scene/concept for the image, including style (e.g., 'A minimalist, brightly lit kitchen with a bowl of fresh oatmeal and berries, overhead shot, soft natural light, warm tones'). Directly represents the pin's visual.
                "Description": "...", // Comprehensive, SEO-rich paragraph (200-500 chars). Integrate 3-5 primary and secondary keywords naturally. Explains the value proposition. Directly relevant.
                "Hashtags": ["#keyword1", "#keyword2", "..."], // Up to 10 relevant, trending, and niche hashtags. Directly related to topic.
                "Alt Text": "..." // Descriptive for accessibility, includes main keywords, max 100 chars. Describes the final image for screen readers. Directly related.
              }},
              // ... 9 more similar objects ...
            ]
            ```

            **Important Instructions:**
            - **Keywords:** Identify primary keywords from the user topic. Sprinkle these throughout the Title, Subtitle, Description, and Alt Text.
            - **Relevance:** Each field must be a direct expansion or facet of the user's initial topic. Do not deviate.
            - **Uniqueness:** While relevant, each of the 10 records should offer a slightly different angle, tip, or focus within the broad user topic.
            - **Image Background Detail:** This field is paramount for generating high-quality visuals. Be explicit about colors, lighting, composition, and style.
            - **Character Limits:** Adhere strictly to provided character limits.
            - **No Markdown outside JSON:** Provide ONLY the JSON array in your response, no conversational text or markdown explanation.

            Now, generate the 10 records for the topic: **"{user_prompt}"**
            """
            response = model.generate_content(prompt_template)
            print("--- Received response from Gemini API ---")

            text_response = response.text.strip()
            print(f"--- Raw Gemini text response (FULL): \n{text_response}\n ---") 

            if text_response.startswith("```json") and text_response.endswith("```"):
                text_response = text_response[7:-3].strip()
            elif not (text_response.startswith("[") and text_response.endswith("]")):
                raise ValueError("Response is not valid JSON and not wrapped in ```json")


            print("--- Attempting to parse JSON ---")
            records = json.loads(text_response)
            print("--- JSON parsed successfully ---")

            for record in records:
                record["Status"] = "Generated - Pending Image"
                record["Image_Url"] = ""
            print("--- Records processed for status and URL ---")
            return records
        except json.JSONDecodeError as e:
            print(f"--- JSON PARSING ERROR (Attempt {attempt + 1}/{retries}): {e} ---")
            if attempt < retries - 1:
                st.warning(f"JSON parsing failed. Retrying... (Attempt {attempt + 1}/{retries})")
                time.sleep(2) 
                continue 
            else:
                st.error(f"JSON parsing failed after {retries} attempts: {e}")
                st.error("The Gemini model returned malformed JSON multiple times. Please check the terminal for the full raw response and prompt quality.")
                return None
        except Exception as e:
            print(f"--- AN ERROR OCCURRED IN generate_seo_records_with_gemini (Attempt {attempt + 1}/{retries}): {e} ---")
            if attempt < retries - 1:
                st.warning(f"Gemini generation failed. Retrying... (Attempt {attempt + 1}/{retries})")
                time.sleep(2) 
                continue 
            else:
                st.error(f"Error generating SEO records after {retries} attempts: {e}")
                st.error("Please check your Gemini API key, prompt quality, and model availability. See terminal for more details.")
                return None
    return None 


# --- Function to generate image using DALL-E 3 ---
def generate_image_with_dalle(image_prompt, index):
    print(f"\n--- Generating image for record {index} with DALL-E ---")
    print(f"--- Original DALL-E prompt for record {index}: {image_prompt} ---")

    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        print(f"--- DALL-E image URL for record {index}: {image_url} ---")

        img_data = requests.get(image_url).content
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        local_image_path = os.path.join(temp_dir, f"pin_image_{index}.png")
        with open(local_image_path, 'wb') as handler:
            handler.write(img_data)
        print(f"--- DALL-E image saved to {local_image_path} ---")
        return local_image_path
    except OpenAI.APIStatusError as e: 
        print(f"--- OPENAI API ERROR for record {index}: {e.status_code} - {e.response} ---")
        st.error(f"Error generating image for record {index}: OpenAI API Error ({e.status_code}). Details in terminal. Check your OpenAI API key and DALL-E prompt content.")
        return None
    except Exception as e:
        print(f"--- GENERAL ERROR IN generate_image_with_dalle for record {index}: {e} ---")
        st.error(f"General error generating image for record {index}: {e}. Check terminal for details.")
        return None

# --- Function to upload image to Google Cloud Storage ---
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    print(f"\n--- Uploading {source_file_name} to GCS bucket {bucket_name} as {destination_blob_name} ---")
    try:
        # Explicitly pass the project ID
        storage_client = storage.Client(project="seopinterestapp") 
        bucket = storage_client.bucket("seo-pinterest-app-images-malko")
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)
        
        # Make the blob publicly viewable (optional, but needed for direct Pinterest linking)
        blob.make_public()

        public_url = blob.public_url
        print(f"--- File {source_file_name} uploaded to {public_url} ---")
        return public_url
    except Exception as e:
        print(f"--- ERROR UPLOADING TO GCS: {e} ---")
        st.error(f"Failed to upload image to GCS: {e}. Check bucket name and GCS authentication/permissions.")
        return None


# --- Main Application Function ---
def main():
    st.set_page_config(page_title="SEO & Pinterest Pin Generator", layout="wide")
    st.title("Automated SEO Content & Pinterest Pin Generator 🚀")
    st.markdown("Enter a topic, and let AI generate SEO-optimized content and Pinterest Pins for you!")

    # --- Debug Info ---
    with st.expander("Show Available Gemini Models (Debug Info)"):
        list_gemini_models()

    # --- Input Fields ---
    st.header("1. Enter Your Topic & Pin Details")

    user_prompt = st.text_area(
        "Main Topic/Keyword for SEO Content (e.g., 'Healthy breakfast ideas for busy professionals')",
        height=100,
        placeholder="Describe the overall theme of your pins..."
    )

    # Pinterest Board and URL inputs
    st.subheader("Pinterest Pin Specifics")
    pinterest_board = st.text_input(
        "Pinterest Board Name/ID (e.g., 'Healthy Recipes Board' or a Board ID)",
        placeholder="Enter the name or ID of your target Pinterest board"
    )
    target_url = st.text_input(
        "Target Website/URL for Pin Link (e.g., '[https://yourwebsite.com/blog/healthy-breakfast](https://yourwebsite.com/blog/healthy-breakfast)')",
        placeholder="Enter the URL your pins should link to"
    )

    # Debug prints to inspect input values
    print(f"\n--- Input Values Check ---")
    print(f"user_prompt: '{user_prompt}' (bool: {bool(user_prompt)})")
    print(f"pinterest_board: '{pinterest_board}' (bool: {bool(pinterest_board)})")
    print(f"target_url: '{target_url}' (bool: {bool(target_url)})")
    print(f"Combined condition (user_prompt and pinterest_board and target_url): {bool(user_prompt and pinterest_board and target_url)}")
    print(f"--- End Input Values Check ---\n")


    # --- Generate Button ---
    if st.button("Generate 10 SEO Records & Pins"):
        print("\n--- Button clicked! ---")
        if user_prompt and pinterest_board and target_url:
            print("--- Inputs are valid! Proceeding to AI generation. ---")
            st.success("Inputs received! Processing your request...")
            st.info("This will take a few moments as AI generates content and images...")

            print("--- About to call generate_seo_records_with_gemini ---")

            generated_records = generate_seo_records_with_gemini(user_prompt)

            if generated_records:
                st.write("---")
                st.subheader("Generated Records (Review & Next Steps)")

                # --- Image Generation Loop ---
                st.subheader("2. Generating Images & Uploading to GCS...")
                
                image_generation_container = st.container()

                # Define your GCS bucket name here!
                GCS_BUCKET_NAME = "seo-pinterest-app-images-malko" 

                for i, record in enumerate(generated_records):
                    image_prompt = record["Image Background"]
                    
                    with image_generation_container:
                        st.info(f"Generating image {i+1}/{len(generated_records)} for: **'{record['Title']}'**")
                        st.text(f"Prompt: {image_prompt[:100]}...")
                        
                        local_image_path = generate_image_with_dalle(image_prompt, i)
                        
                        if local_image_path:
                            # Upload to GCS
                            destination_blob_name = f"pinterest_pins/{os.path.basename(local_image_path)}"
                            gcs_public_url = upload_to_gcs(GCS_BUCKET_NAME, local_image_path, destination_blob_name)

                            if gcs_public_url:
                                record["Image_Url"] = gcs_public_url # Store GCS public URL
                                record["Status"] = "Image Uploaded to GCS"
                                st.image(gcs_public_url, caption=f"Uploaded Image {i+1}", width=200) # Display from GCS URL
                                st.success(f"Image {i+1} generated and uploaded to GCS successfully! URL: {gcs_public_url}")
                                # Clean up local image after upload
                                try:
                                    os.remove(local_image_path)
                                    print(f"--- Removed local image: {local_image_path} ---")
                                except OSError as e:
                                    print(f"--- Error removing local image {local_image_path}: {e} ---")
                            else:
                                record["Status"] = "GCS Upload Failed"
                                st.error(f"Failed to upload image {i+1} to GCS.")
                                # Keep local image for debugging if GCS fails
                                record["Image_Url"] = local_image_path # Store local path if GCS fails
                        else:
                            record["Status"] = "Image Generation Failed"
                            st.error(f"Failed to generate image {i+1}.")
                
                with image_generation_container:
                    st.success("Image generation and GCS upload process complete for all records!")
                
                df = pd.DataFrame(generated_records)
                st.dataframe(df, height=400) 

                st.info("Next: We will add text overlay (if desired) and then post to Pinterest using the GCS image URLs.")
            else:
                st.error("Failed to generate SEO records. Cannot proceed with image generation/upload. Check terminal for specific errors.")

        else:
            print("--- Inputs are NOT valid! Displaying warning. ---")
            st.warning("Please fill in all input fields.")

if __name__ == "__main__":
    main()