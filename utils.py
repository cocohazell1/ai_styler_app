# --- START OF FILE utils.py ---

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import os
import mediapipe as mp

# --- MediaPipe 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
# Ensure refine_landmarks=True for more detailed landmarks, especially around eyes/lips
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 랜드마크 인덱스 (더 상세하게 정의) ---
# 주: 정확한 메이크업 영역은 인덱스 조합과 마스크 생성 방식에 따라 달라짐
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 402, 317, 14, 87, 178, 88, 95, 78]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95] # 내부 립 경계 (참고용)

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] # 눈꺼풀 주변
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# 블러셔 영역 추정 (광대뼈 부근 랜드마크 활용 - 예시)
LEFT_CHEEK = [117, 118, 119, 101, 147, 205, 213, 135, 136] # 왼쪽 광대뼈 주변 (조금 더 넓게)
RIGHT_CHEEK = [346, 347, 348, 330, 376, 425, 433, 364, 365] # 오른쪽 광대뼈 주변 (조금 더 넓게)


def load_image(image_file):
    """이미지 파일을 PIL Image 객체로 로드하고 RGB로 변환"""
    if image_file is None: return None
    try:
        img = Image.open(image_file).convert('RGB')
        # # 이미지 방향 보정 (EXIF 정보 사용 - 선택적, 필요시 주석 해제)
        # from PIL import ExifTags
        # try:
        #     for orientation in ExifTags.TAGS.keys():
        #         if ExifTags.TAGS[orientation]=='Orientation':
        #             break
        #     exif=dict(img._getexif().items())
        #     if exif[orientation] == 3: img=img.rotate(180, expand=True)
        #     elif exif[orientation] == 6: img=img.rotate(270, expand=True)
        #     elif exif[orientation] == 8: img=img.rotate(90, expand=True)
        # except (AttributeError, KeyError, IndexError, TypeError):
        #     # cases: image doesn't have getexif or other errors
        #     pass
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def pil_to_cv2(pil_img):
    """PIL(RGB) -> OpenCV(BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    """OpenCV(BGR) -> PIL(RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


# ***** ADD THIS FUNCTION *****
def detect_face_landmarks(img_pil):
    """
    Detects face landmarks using MediaPipe Face Mesh.

    Args:
        img_pil (PIL.Image): Input image in PIL format (RGB).

    Returns:
        tuple: A tuple containing:
            - results: The Face Mesh results object from MediaPipe.
                       (None if no face detected or error occurred)
            - img_width: Width of the input image.
            - img_height: Height of the input image.
                       (Returns 0, 0 if input is invalid)
    """
    if img_pil is None:
        print("Error: Input image is None for landmark detection.")
        return None, 0, 0

    try:
        # Convert PIL Image to NumPy array (RGB)
        img_rgb = np.array(img_pil.convert('RGB'))
        img_height, img_width, _ = img_rgb.shape

        # Process the image and find face landmarks
        # Make sure 'face_mesh' is initialized globally in utils.py
        results = face_mesh.process(img_rgb)

        # Check if landmarks were detected
        # if not results.multi_face_landmarks:
        #     print("Warning: No face landmarks detected in the image.")
            # No need to explicitly return None here, results object handles it

        return results, img_width, img_height

    except Exception as e:
        print(f"Error during face landmark detection: {e}")
        # Attempt to return dimensions even on error, if possible
        try:
             w, h = img_pil.size # Correct order for PIL size
        except:
            w, h = 0, 0
        return None, w, h
# ***** END OF ADDED FUNCTION *****


def get_landmark_points(landmarks, indices, img_width, img_height):
    """랜드마크 결과에서 특정 인덱스의 좌표 리스트 추출"""
    points = []
    valid_indices_count = 0
    # Check if landmarks and the specific landmark list exist
    if landmarks and landmarks.landmark:
        num_landmarks = len(landmarks.landmark)
        for idx in indices:
            if 0 <= idx < num_landmarks:
                lm = landmarks.landmark[idx]
                # 좌표가 [0, 1] 범위 내에 있는지 확인 (정규화된 좌표)
                if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                    # 이미지 경계 내 절대 좌표로 변환
                    abs_x = int(lm.x * img_width)
                    abs_y = int(lm.y * img_height)
                    # 혹시 모를 반올림 오류로 경계 벗어나는 것 방지
                    abs_x = max(0, min(abs_x, img_width - 1))
                    abs_y = max(0, min(abs_y, img_height - 1))
                    points.append((abs_x, abs_y))
                    valid_indices_count += 1
            # else:
                # print(f"Warning: Landmark index {idx} out of range ({num_landmarks} landmarks available).")
    # 유효한 포인트가 최소 3개 이상이어야 폴리곤을 그릴 수 있음
    return points if valid_indices_count >= 3 else []


def hex_to_rgb(hex_color):
    """Hex 색상 코드를 RGB 튜플로 변환"""
    hex_color = hex_color.lstrip('#')
    try:
        # Ensure length is 6
        if len(hex_color) == 3: # Allow shorthand hex (e.g., #F00)
            hex_color = "".join([c*2 for c in hex_color])
        if len(hex_color) != 6:
             raise ValueError("Invalid hex color length")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError as e:
        print(f"Invalid hex color: '{hex_color}'. Using default red. Error: {e}")
        return (255, 0, 0) # 오류 시 기본 색상 반환

def apply_makeup(img_pil, makeup_options):
    """얼굴 랜드마크 기반으로 다양한 메이크업 효과 적용"""
    output_img_pil = img_pil.copy()
    intensity_factor = makeup_options.get('intensity', 0.5) # 전체 강도

    # --- 1. 얼굴 랜드마크 감지 ---
    # Call the newly added function
    landmarks_results, img_width, img_height = detect_face_landmarks(output_img_pil)

    # Check if landmarks were detected *and* if the results object exists
    if not landmarks_results or not landmarks_results.multi_face_landmarks:
        print("랜드마크 감지 실패. 메이크업을 적용할 수 없습니다.")
        # # Optionally apply a simple filter even without landmarks
        # enhancer = ImageEnhance.Color(output_img_pil)
        # output_img_pil = enhancer.enhance(1.0 + 0.1 * intensity_factor)
        # enhancer = ImageEnhance.Contrast(output_img_pil)
        # output_img_pil = enhancer.enhance(1.0 + 0.05 * intensity_factor)
        return output_img_pil, False # 실패 플래그 반환

    # Proceed only if landmarks are found
    face_landmarks = landmarks_results.multi_face_landmarks[0] # 첫 번째 감지된 얼굴 사용

    # --- 2. 메이크업 효과 적용을 위한 오버레이 준비 ---
    overlay = Image.new('RGBA', output_img_pil.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # --- 3. 각 메이크업 요소 적용 ---
    applied_effects = [] # 어떤 효과가 적용되었는지 기록

    # 💄 입술 (Lips)
    if makeup_options.get('apply_lips', False):
        # Use LIPS_OUTER for the boundary
        lip_points = get_landmark_points(face_landmarks, LIPS_OUTER, img_width, img_height)
        if lip_points:
            lip_color_rgb = hex_to_rgb(makeup_options.get('lip_color', '#E64E6B'))
            lip_intensity = makeup_options.get('lip_intensity', intensity_factor)
            # Adjust alpha calculation (e.g., less transparency for lips)
            lip_fill_color = lip_color_rgb + (int(255 * lip_intensity * 0.8),)
            overlay_draw.polygon(lip_points, fill=lip_fill_color)
            applied_effects.append("입술")
        else:
            print("Warning: Not enough lip points detected to apply lip makeup.")

    # ✨ 아이섀도우 (Eyeshadow)
    if makeup_options.get('apply_eyeshadow', False):
        # Define points slightly above the eye for shadow
        # This requires more sophisticated landmark combinations or convex hull around eye area
        # Simple approach using eye boundaries:
        left_eye_points = get_landmark_points(face_landmarks, LEFT_EYE, img_width, img_height)
        right_eye_points = get_landmark_points(face_landmarks, RIGHT_EYE, img_width, img_height)

        if left_eye_points and right_eye_points:
            eye_color_rgb = hex_to_rgb(makeup_options.get('eyeshadow_color', '#8A5A94'))
            eye_intensity = makeup_options.get('eyeshadow_intensity', intensity_factor)
            # Adjust alpha for eyeshadow (might need less transparency than lips)
            eye_fill_color = eye_color_rgb + (int(255 * eye_intensity * 0.6),)

            # Create a temporary mask for blurring the eyeshadow
            eye_mask = Image.new('RGBA', overlay.size, (0,0,0,0))
            eye_draw = ImageDraw.Draw(eye_mask)

            # Draw polygons on the temporary mask
            eye_draw.polygon(left_eye_points, fill=eye_fill_color)
            eye_draw.polygon(right_eye_points, fill=eye_fill_color)

            # Blur the eyeshadow mask for softer edges
            blur_radius = max(5, int(img_width * 0.02)) # Adjust blur radius as needed
            eye_mask_blurred = eye_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Composite the blurred eyeshadow onto the main overlay
            overlay = Image.alpha_composite(overlay, eye_mask_blurred)
            applied_effects.append("아이섀도우")
        else:
            print("Warning: Not enough eye points detected to apply eyeshadow.")

    # 😊 블러셔 (Blush) - Gaussian blur approach
    if makeup_options.get('apply_blush', False):
        left_cheek_points = get_landmark_points(face_landmarks, LEFT_CHEEK, img_width, img_height)
        right_cheek_points = get_landmark_points(face_landmarks, RIGHT_CHEEK, img_width, img_height)

        if left_cheek_points and right_cheek_points:
            blush_color_rgb = hex_to_rgb(makeup_options.get('blush_color', '#F08080'))
            blush_intensity = makeup_options.get('blush_intensity', intensity_factor)
            # Blush alpha - typically more subtle
            blush_alpha = int(255 * blush_intensity * 0.45)
            blush_color_rgba = blush_color_rgb + (blush_alpha,)

            # Calculate approximate cheek centers
            left_center = np.mean(left_cheek_points, axis=0).astype(int)
            right_center = np.mean(right_cheek_points, axis=0).astype(int)

            # Determine blush radius based on face size/intensity
            # Distance between eyes can be a proxy for face scale
            try:
                left_eye_center = np.mean(get_landmark_points(face_landmarks, LEFT_EYE, img_width, img_height), axis=0)
                right_eye_center = np.mean(get_landmark_points(face_landmarks, RIGHT_EYE, img_width, img_height), axis=0)
                eye_dist = np.linalg.norm(left_eye_center - right_eye_center)
                radius = int(eye_dist * 0.4 * blush_intensity + img_width * 0.03) # Combine factors
            except: # Fallback if eye points fail
                radius = int(img_width * 0.06 * blush_intensity + 10)
            radius = max(10, radius) # Minimum radius

            # Create a blush layer and apply Gaussian blur
            blush_layer = Image.new('RGBA', output_img_pil.size, (0, 0, 0, 0))
            blush_draw = ImageDraw.Draw(blush_layer)

            # Draw filled ellipses at cheek centers (adjust size/shape as needed)
            # Make ellipses slightly oval vertically
            blush_draw.ellipse((left_center[0]-radius, left_center[1]-int(radius*1.2), left_center[0]+radius, left_center[1]+int(radius*1.2)), fill=blush_color_rgba)
            blush_draw.ellipse((right_center[0]-radius, right_center[1]-int(radius*1.2), right_center[0]+radius, right_center[1]+int(radius*1.2)), fill=blush_color_rgba)

            # Apply significant blur for a soft effect
            blur_radius_blush = radius * 2.0 # Larger blur for blush
            blush_layer_blurred = blush_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius_blush))

            # Composite the blurred blush onto the main overlay
            overlay = Image.alpha_composite(overlay, blush_layer_blurred)
            applied_effects.append("블러셔")
        else:
            print("Warning: Not enough cheek points detected to apply blush.")


    # --- 4. 생성된 오버레이 합성 ---
    if applied_effects:
        try:
            # The overlay already contains blurred elements where needed
            # No need for an overall blur here if elements are blurred individually
            output_img_pil = output_img_pil.convert('RGBA')
            output_img_pil = Image.alpha_composite(output_img_pil, overlay)
            output_img_pil = output_img_pil.convert('RGB')
        except Exception as e:
            print(f"Error applying makeup overlay: {e}")
            # Return original image if compositing fails
            return img_pil.copy(), False

    # --- 5. 피부 보정 등 추가 효과 (선택적 - placeholder) ---
    # if makeup_options.get('skin_smoothing', False):
         # Simple skin smoothing (optional, can be slow)
         # img_cv = pil_to_cv2(output_img_pil)
         # smoothed_cv = cv2.bilateralFilter(img_cv, d=7, sigmaColor=50, sigmaSpace=50)
         # output_img_pil = cv2_to_pil(smoothed_cv)
         # pass

    return output_img_pil, True # 성공 플래그 반환


def apply_makeup_transfer(face_pil, style_pil):
    """참조 스타일 이미지의 색감을 얼굴 이미지에 전송 (개선된 색상 전송)"""
    if face_pil is None or style_pil is None: return face_pil, False

    # --- 1. 얼굴 랜드마크 감지 (얼굴 영역 마스크 생성용) ---
    landmarks_results, img_width, img_height = detect_face_landmarks(face_pil)
    if not landmarks_results or not landmarks_results.multi_face_landmarks:
        print("랜드마크 감지 실패. 메이크업 전송을 위한 얼굴 영역을 찾을 수 없습니다.")
        # 전체 이미지에 색상 전송 시도 (대체 옵션)
        try:
            # Make sure apply_color_transfer handles potential errors
            transferred_face = apply_color_transfer(style_pil, face_pil)
            return transferred_face, True
        except Exception as e:
            print(f"전체 이미지 색상 전송 실패: {e}")
            return face_pil, False

    face_landmarks = landmarks_results.multi_face_landmarks[0]

    # --- 2. 얼굴 영역 마스크 생성 (더 정교하게) ---
    # Get all landmark points for convex hull
    all_points = []
    if face_landmarks.landmark:
         num_landmarks = len(face_landmarks.landmark)
         all_points = get_landmark_points(face_landmarks, list(range(num_landmarks)), img_width, img_height)

    if not all_points:
        print("랜드마크 포인트 추출 실패. 마스크를 생성할 수 없습니다.")
        # Fallback to full image transfer
        try:
            transferred_face = apply_color_transfer(style_pil, face_pil)
            return transferred_face, True
        except Exception as e:
            print(f"전체 이미지 색상 전송 실패 (마스크 생성 불가): {e}")
            return face_pil, False


    final_mask = None
    try:
        # Create convex hull from all points
        hull = cv2.convexHull(np.array(all_points), returnPoints=True)
        hull_points = [tuple(p[0]) for p in hull]

        if len(hull_points) > 2:
            face_mask = Image.new('L', face_pil.size, 0)
            face_draw = ImageDraw.Draw(face_mask)
            face_draw.polygon(hull_points, fill=255)

            # Dilate and blur the mask for softer edges
            # Adjust kernel sizes and iterations for desired softness
            kernel_dilate = np.ones((15,15), np.uint8) # Dilation kernel
            kernel_blur = (31, 31)                    # Gaussian blur kernel size

            mask_cv = np.array(face_mask)
            mask_dilated = cv2.dilate(mask_cv, kernel_dilate, iterations=3) # Increase iterations for more expansion
            mask_blurred = cv2.GaussianBlur(mask_dilated, kernel_blur, 0)
            final_mask = Image.fromarray(mask_blurred)
        else:
             print("Convex Hull 생성 실패. 얼굴 마스크를 만들 수 없습니다.")

    except Exception as e:
        print(f"얼굴 마스크 생성 중 오류: {e}")
        final_mask = None # Ensure mask is None on error

    # --- 3. 색상 전송 적용 ---
    try:
        # Resize style image to match face image size for color stats
        style_resized = style_pil.resize(face_pil.size, Image.Resampling.LANCZOS)
        # Apply color transfer
        transferred_face = apply_color_transfer(style_resized, face_pil)
    except Exception as e:
        print(f"색상 전송 중 오류: {e}")
        return face_pil, False # Return original face on color transfer error

    # --- 4. 마스크를 이용해 원본과 합성 ---
    if final_mask:
        try:
            # Ensure both images are RGB before compositing with mask
            output_img = Image.composite(transferred_face.convert('RGB'), face_pil.convert('RGB'), final_mask)
            return output_img, True
        except Exception as e:
            print(f"마스크 합성 중 오류: {e}")
            # Fallback to returning the transferred face if composite fails
            return transferred_face, True
    else:
        # Mask failed, return full transfer result
        print("Warning: 얼굴 마스크 없이 전체 이미지에 색상 전송 결과 반환.")
        return transferred_face, True


def apply_color_transfer(source_pil, target_pil):
    """OpenCV 컬러 전송 (Lab 색상 공간) - 소스 이미지 색감을 타겟에 적용"""
    try:
        # Ensure images are RGB PIL objects
        source_pil = source_pil.convert('RGB')
        target_pil = target_pil.convert('RGB')

        # Convert to OpenCV BGR format
        source = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
        target = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)

        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate statistics
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)

        # Avoid division by zero or very small numbers
        target_std[target_std < 1e-6] = 1e-6
        source_std[source_std < 1e-6] = 1e-6 # Also check source_std

        # Apply color transfer equation
        # Subtract target mean, scale by std dev ratio, add source mean
        l_target, a_target, b_target = cv2.split(target_lab)
        l_mean_s, a_mean_s, b_mean_s = source_mean.flatten()
        l_std_s, a_std_s, b_std_s = source_std.flatten()
        l_mean_t, a_mean_t, b_mean_t = target_mean.flatten()
        l_std_t, a_std_t, b_std_t = target_std.flatten()

        l_transfer = ((l_target - l_mean_t) * (l_std_s / l_std_t)) + l_mean_s
        a_transfer = ((a_target - a_mean_t) * (a_std_s / a_std_t)) + a_mean_s
        b_transfer = ((b_target - b_mean_t) * (b_std_s / b_std_t)) + b_mean_s

        # Merge channels and clip values
        transferred_lab = cv2.merge([l_transfer, a_transfer, b_transfer])
        transferred_lab = np.clip(transferred_lab, 0, 255) # Clip values to valid range

        # Convert back to BGR uint8
        result_bgr = cv2.cvtColor(transferred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Convert back to PIL RGB
        return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

    except cv2.error as e:
        print(f"OpenCV error during color transfer: {e}")
        # Return target image if specific OpenCV error occurs
        return target_pil
    except Exception as e:
        print(f"General error during color transfer: {e}")
        # Re-raise the exception to be caught by the caller if needed
        # or return the original target image as a fallback
        return target_pil
        # raise e


def apply_fashion_filter(img_pil, style="casual", intensity=0.7):
    """선택된 스타일과 강도에 따라 패션 필터 효과 적용 (개선된 세피아)"""
    img = img_pil.copy()
    if intensity == 0: return img # 강도가 0이면 원본 반환
    try:
        if style == "casual":
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.0 + 0.15 * intensity)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.0 + 0.1 * intensity)
        elif style == "vintage":
            # Apply sepia first
            img_array = np.array(img.convert('RGB')) # Ensure RGB
            # Normalize intensity for sepia matrix calculation
            normalized_intensity = max(0.0, min(1.0, intensity)) # Clamp between 0 and 1

            # Sepia matrix - values closer to identity matrix at lower intensity
            identity = np.identity(3)
            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            # Blend sepia kernel with identity based on intensity
            transform_matrix = identity * (1 - normalized_intensity) + sepia_kernel * normalized_intensity

            # Apply the transformation
            sepia_img_array = cv2.transform(img_array, transform_matrix)
            sepia_img_array = np.clip(sepia_img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(sepia_img_array)

            # Apply other vintage effects after sepia
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.0 - 0.2 * intensity) # Slightly less desaturation
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.0 + 0.15 * intensity)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.0 - 0.05 * intensity) # Slight darkening

        elif style == "elegant":
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.0 + 0.25 * intensity)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.0 + 0.4 * intensity)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.0 + 0.05 * intensity)
            # Slightly reduce color saturation for elegance
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.0 - 0.1 * intensity)
        elif style == "monochrome":
            # Convert to grayscale using Pillow's L mode
            img = img.convert('L')
            # Convert back to RGB for consistency if needed downstream,
            # or keep as L if the rest of the pipeline handles it.
            img = img.convert('RGB')
            # Adjust contrast on the monochrome image
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.0 + 0.3 * intensity)
        return img
    except Exception as e:
        print(f"Error applying fashion filter '{style}': {e}")
        return img_pil # Return original on error


def change_clothing_color(clothing_img_pil, target_color_hex):
    """의상 이미지의 색상을 변경 (HSV 기반 - 투명도 유지)"""
    if clothing_img_pil is None: return None
    try:
        target_color_rgb = hex_to_rgb(target_color_hex)
        # Convert target RGB to HSV for Hue comparison
        target_hsv = cv2.cvtColor(np.uint8([[target_color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        target_h, target_s, target_v = target_hsv

        # Ensure input image has Alpha channel
        img_rgba = clothing_img_pil.convert("RGBA")
        img_cv_rgba = np.array(img_rgba)

        # Separate RGB and Alpha channels
        img_cv_rgb = img_cv_rgba[:,:,:3]
        alpha_channel = img_cv_rgba[:,:,3]

        # Convert RGB to HSV
        img_hsv = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2HSV)
        original_h, original_s, original_v = cv2.split(img_hsv)

        # Create a mask based on Alpha and Saturation
        # Adjust thresholds as needed for different clothing items
        alpha_threshold = 50  # Pixels with low alpha are ignored
        saturation_threshold = 25 # Ignore grayscale pixels (low saturation)
        value_threshold_low = 20   # Ignore very dark pixels
        value_threshold_high = 235 # Ignore very bright pixels (optional)

        mask = (alpha_channel > alpha_threshold) & \
               (original_s > saturation_threshold) & \
               (original_v > value_threshold_low) # & \
               # (original_v < value_threshold_high)

        # --- Hue Change ---
        # Replace Hue in masked area with target Hue
        changed_h = np.where(mask, target_h, original_h)

        # --- Saturation Adjustment ---
        # Option 1: Set to target saturation (can look unnatural)
        # changed_s = np.where(mask, target_s, original_s)
        # Option 2: Scale original saturation towards target saturation
        # Factor determines how much influence the target saturation has
        sat_factor = 0.7 # 0 = original saturation, 1 = target saturation
        scaled_s = np.clip(original_s * (1 - sat_factor) + target_s * sat_factor, 0, 255)
        changed_s = np.where(mask, scaled_s, original_s)
        # Option 3: Adjust based on ratio (more complex)
        # target_s_norm = target_s / 255.0
        # original_s_norm = original_s / 255.0
        # # Adjust saturation more aggressively if target is colorful
        # s_adjust_factor = 1.0 + (target_s_norm - 0.5) * 0.5
        # changed_s = np.where(mask, np.clip(original_s * s_adjust_factor, 0, 255), original_s)


        # --- Value (Brightness) Adjustment ---
        # Option 1: Set to target value (can lose detail)
        # changed_v = np.where(mask, target_v, original_v)
        # Option 2: Scale original value towards target value
        val_factor = 0.6 # 0 = original value, 1 = target value
        scaled_v = np.clip(original_v * (1 - val_factor) + target_v * val_factor, 0, 255)
        changed_v = np.where(mask, scaled_v, original_v)
        # Option 3: Adjust brightness relative to original (preserve shadows/highlights)
        # Calculate average value in the mask
        # avg_v_original = np.mean(original_v[mask]) if np.any(mask) else 128
        # avg_v_target = target_v
        # v_diff = avg_v_target - avg_v_original
        # # Apply the difference, clipping to 0-255
        # changed_v = np.where(mask, np.clip(original_v + v_diff * 0.8, 0, 255), original_v)


        # Merge the modified HSV channels
        final_hsv = cv2.merge([changed_h, changed_s.astype(np.uint8), changed_v.astype(np.uint8)])

        # Convert back to RGB
        result_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

        # Combine with the original alpha channel
        result_rgba = np.dstack((result_rgb, alpha_channel))

        return Image.fromarray(result_rgba)

    except ValueError as e:
         print(f"Invalid hex color format for clothing: {target_color_hex}. Error: {e}")
         return clothing_img_pil # Return original on hex error
    except cv2.error as e:
        print(f"OpenCV error changing clothing color: {e}")
        return clothing_img_pil # Return original on OpenCV error
    except Exception as e:
        print(f"Error changing clothing color: {e}")
        return clothing_img_pil # Return original on other errors


def virtual_try_on(person_img_pil, clothing_img_pil, position=(0, 0), scale=1.0):
    """가상 의상 입히기 (위치/크기 조절, 알파 블렌딩 개선)"""
    if person_img_pil is None or clothing_img_pil is None:
        print("Error: Input image(s) missing for virtual try-on.")
        return person_img_pil.convert('RGB') if person_img_pil else None

    try:
        # Ensure person image is RGBA for compositing
        person_rgba = person_img_pil.convert("RGBA")
        p_width, p_height = person_rgba.size

        # Ensure clothing image is RGBA
        clothing_rgba = clothing_img_pil.convert("RGBA")
        c_width, c_height = clothing_rgba.size

        # --- Scale Clothing ---
        new_c_width = int(c_width * scale)
        new_c_height = int(c_height * scale)

        # Prevent zero or negative dimensions after scaling
        if new_c_width <= 0 or new_c_height <= 0:
            print(f"Warning: Invalid clothing scale resulted in zero/negative size ({new_c_width}x{new_c_height}). Skipping try-on.")
            return person_img_pil.convert('RGB') # Return original person image

        # Use LANCZOS for high-quality resizing
        clothing_resized = clothing_rgba.resize((new_c_width, new_c_height), Image.Resampling.LANCZOS)

        # --- Position Clothing ---
        # position[0] = X (left offset), position[1] = Y (top offset)
        paste_x = int(position[0])
        paste_y = int(position[1])

        # --- Create Background ---
        # Start with the original person image
        result_img = person_rgba.copy()

        # --- Prepare Mask ---
        # Get the alpha channel of the resized clothing item
        try:
            # clothing_resized.split() might return (R, G, B) if no alpha exists
            if clothing_resized.mode == 'RGBA':
                mask = clothing_resized.split()[3]
            else:
                # If clothing has no alpha, create a fully opaque mask
                print("Warning: Clothing image does not have an alpha channel. Using opaque mask.")
                mask = Image.new('L', clothing_resized.size, 255)
        except IndexError:
            print("Warning: Could not get alpha channel from clothing. Using opaque mask.")
            mask = Image.new('L', clothing_resized.size, 255)
        except Exception as e:
            print(f"Error getting clothing mask: {e}. Using opaque mask.")
            mask = Image.new('L', clothing_resized.size, 255)


        # --- Composite Images ---
        # Paste the clothing onto the person image using the alpha mask
        # The paste coordinates define the top-left corner of the clothing item
        # PIL's paste handles pixels outside the bounds gracefully (they are ignored)
        result_img.paste(clothing_resized, (paste_x, paste_y), mask)

        # Convert final result back to RGB
        return result_img.convert('RGB')

    except Exception as e:
        print(f"Error during virtual try-on: {e}")
        # Return the original person image in case of errors
        return person_img_pil.convert('RGB')


def create_assets_folder():
    """앱 실행에 필요한 에셋 폴더 생성"""
    folders = ["assets", "assets/clothes", "assets/makeup_styles", "assets/examples", "user_gallery"]
    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                print(f"폴더 생성됨: {folder}")
            except OSError as e:
                print(f"폴더 생성 실패: {folder}, 오류: {e}")

# --- END OF FILE utils.py ---