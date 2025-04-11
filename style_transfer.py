import cv2
import numpy as np
from PIL import Image, ImageFilter
import requests
from io import BytesIO
import os

# --- 샘플 의상 이미지 URL 정의 ---
CLOTHING_URLS = {
    "casual_tshirt": "https://www.publicdomainpictures.net/pictures/320000/nahled/t-shirt-transparent.png",
    "formal_dress": "https://www.publicdomainpictures.net/pictures/260000/nahled/dress-red-womens-fashion.jpg",
    "sporty_jacket": "https://cdn.pixabay.com/photo/2017/03/05/15/29/sport-2119296_960_720.png",
    "vintage_blouse": "https://cdn.pixabay.com/photo/2017/08/10/02/05/vintage-2617091_960_720.jpg",
}

# --- 메이크업 스타일 설명 (참고용) ---
MAKEUP_STYLES_INFO = {
    "Natural_Daily": {"description": "자연스러운 데일리 룩"},
    "Classic_Glamour": {"description": "클래식 글래머 룩 (레드립)"},
    "Smoky_Eye": {"description": "스모키 아이 룩"},
    "Peach_Coral": {"description": "피치/코랄 룩"},
    "Bold_Lip": {"description": "볼드 립 룩"}
}


def download_image_from_url(url, timeout=10):
    """URL에서 이미지 다운로드 (이전과 동일)"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type')
        if not content_type or not content_type.startswith('image/'):
             print(f"Warning: URL might not point to an image. Content-Type: {content_type}, URL: {url}")
        img = Image.open(BytesIO(response.content))
        return img.convert('RGBA')
    except requests.exceptions.Timeout:
        print(f"Image download timeout: {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Image download failed: {url}, Error: {e}")
        return None
    except Exception as e:
        print(f"Error processing image from URL: {url}, Error: {e}")
        return None

def load_images_from_folder(folder_path):
    """지정된 폴더에서 이미지 파일들을 로드하여 딕셔너리로 반환"""
    images = {}
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder not found - {folder_path}")
        return images
    print(f"Loading images from: {folder_path}")
    loaded_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            style_name = os.path.splitext(filename)[0] # 파일명이 스타일 이름
            try:
                img_path = os.path.join(folder_path, filename)
                if os.path.getsize(img_path) == 0:
                    print(f"Warning: Skipping empty file - {filename}")
                    continue
                img = Image.open(img_path).convert('RGBA') # RGBA로 로드
                images[style_name] = img
                loaded_count += 1
            except FileNotFoundError:
                 print(f"File not found: {img_path}")
            except Exception as e:
                print(f"Failed to load image: {filename}, Error: {e}")
    print(f"Loaded {loaded_count} images from {folder_path}.")
    return images


def prepare_clothing_samples(use_local=True, local_dir="assets/clothes", fallback_to_url=True):
    """샘플 의상 이미지 준비 (로컬 우선)"""
    clothing_images = {}
    if use_local:
        clothing_images = load_images_from_folder(local_dir)

    if not clothing_images and fallback_to_url:
        print("No local clothing samples found. Attempting to download from URLs...")
        loaded_count = 0
        for style, url in CLOTHING_URLS.items():
             img = download_image_from_url(url)
             if img:
                 clothing_images[style] = img
                 loaded_count +=1
        print(f"Downloaded {loaded_count} clothing samples from URLs.")

    return clothing_images

def prepare_makeup_style_samples(local_dir="assets/makeup_styles"):
    """메이크업 스타일 참조 이미지 로드"""
    return load_images_from_folder(local_dir)

# --- 색상 전송 및 메이크업 전송 함수는 utils.py로 이동/통합 ---