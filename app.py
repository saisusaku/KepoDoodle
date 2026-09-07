import os
import time
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Menggunakan Absolute Path agar aman di server cloud seperti Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
IMAGE_FOLDER = os.path.join(BASE_DIR, "image")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Dictionary untuk menyimpan timestamp terakhir akses per IP (Cooldown 2 Jam = 7200 detik)
IP_COOLDOWN = {}
COOLDOWN_TIME = 7200  # 2 jam dalam detik

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/generate", methods=["POST"])
def generate_doodle():
    # Ambil IP Client (mendukung proxy jika di-deploy di cloud)
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    current_time = time.time()

    # Cek apakah IP terkena cooldown
    if client_ip in IP_COOLDOWN:
        elapsed_time = current_time - IP_COOLDOWN[client_ip]
        if elapsed_time < COOLDOWN_TIME:
            remaining_seconds = int(COOLDOWN_TIME - elapsed_time)
            remaining_minutes = remaining_seconds // 60
            return jsonify({
                "error": f"Batas tercapai! Anda dapat membuat video baru lagi dalam {remaining_minutes} menit ke depan (1 video per 2 jam per perangkat/IP)."
            }), 429

    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    orientation = request.form.get("orientation", "horizontal").lower()

    input_filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)

    base_name = os.path.splitext(input_filename)[0]
    video_filename = f"{base_name}_{orientation}_sequential_color_doodle.mp4"
    video_path = os.path.join(OUTPUT_FOLDER, video_filename)

    try:
        if orientation == "potret" or orientation == "portrait":
            width, height = 720, 1280
        else:
            width, height = 1280, 720

        fps = 30

        # --- LOAD GAMBAR TANGAN DENGAN ABSOLUTE PATH ---
        hand_path = os.path.join(IMAGE_FOLDER, "tangan.png")
        hand_img = None
        
        if os.path.exists(hand_path):
            # Gunakan IMREAD_UNCHANGED agar channel transparansi (PNG alpha) ikut terbaca
            hand_img = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
            if hand_img is not None:
                # Jika gambar tidak memiliki alpha channel (misal JPG/PNG biasa tanpa transparan), tambahkan alpha channel buatan
                if hand_img.shape[2] == 3:
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2BGRA)
                
                h_h, h_w = hand_img.shape[:2]
                new_w = 450
                new_h = int(h_h * (new_w / h_w))
                hand_img = cv2.resize(hand_img, (new_w, new_h))
        else:
            print(f"Peringatan: File tangan.png tidak ditemukan di path: {hand_path}")

        img_raw = cv2.imread(input_path)
        if img_raw is None:
            return jsonify({"error": "Gagal membaca file gambar."}), 400
        
        # Crop / Fit sesuai orientasi
        h_orig, w_orig = img_raw.shape[:2]
        target_aspect = width / height
        orig_aspect = w_orig / h_orig

        if orig_aspect > target_aspect:
            new_w = int(h_orig * target_aspect)
            start_x = (w_orig - new_w) // 2
            img_cropped = img_raw[:, start_x:start_x + new_w]
        else:
            new_h = int(w_orig / target_aspect)
            start_y = (h_orig - new_h) // 2
            img_cropped = img_raw[start_y:start_y + new_h, :]

        img = cv2.resize(img_cropped, (width, height))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contour_paths = []
        cluster_list = []

        def sort_points_snake(pts):
            if not pts:
                return []
            rows = {}
            for x, y in pts:
                if y not in rows:
                    rows[y] = []
                rows[y].append((x, y))
            final_pts = []
            reverse = False
            for y in sorted(rows.keys()):
                row_pts = sorted(rows[y], key=lambda p: p[0], reverse=reverse)
                final_pts.extend(row_pts)
                reverse = not reverse
            return final_pts

        # 1. Tahap 1: Ambil Outline
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        for cnt in contours:
            if len(cnt) > 2:
                pts = [tuple(pt[0]) for pt in cnt]
                contour_paths.append(pts)

        # 2. Proses Area Warna & Hitam
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        flood_img = img.copy()
        cv2.floodFill(flood_img, flood_mask, (0, 0), (0, 0, 0), (10, 10, 10), (10, 10, 10), cv2.FLOODFILL_FIXED_RANGE)
        
        bg_color = img[0, 0]
        is_not_outer_bg = np.any(np.abs(img.astype(int) - bg_color.astype(int)) > 15, axis=2)
        
        black_mask = (gray_img < 50) & (img[:, :, 0] < 50) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50)
        valid_black_mask = black_mask & is_not_outer_bg
        
        y_b, x_b = np.where(valid_black_mask)
        if len(x_b) > 0:
            black_pts = list(zip(x_b, y_b))
            cluster_list.append({
                "color": np.array([0, 0, 0], dtype=np.uint8),
                "points": sort_points_snake(black_pts),
                "brightness": 0
            })

        white_inner_mask = is_not_outer_bg & (gray_img > 230) & (img[:, :, 0] > 230) & (img[:, :, 1] > 230) & (img[:, :, 2] > 230)
        colored_pixel_mask = is_not_outer_bg & ~black_mask & ~white_inner_mask

        y_indices, x_indices = np.where(colored_pixel_mask)
        color_data = img[y_indices, x_indices] if len(y_indices) > 0 else np.zeros((1, 3), dtype=np.uint8)

        if len(color_data) > 0:
            data_f = np.float32(color_data)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 3
            _, labels, centers = cv2.kmeans(data_f, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)

            for i in range(k):
                sub_mask = (labels.flatten() == i)
                sub_x = x_indices[sub_mask]
                sub_y = y_indices[sub_mask]
                if len(sub_x) > 0:
                    pts = list(zip(sub_x, sub_y))
                    mean_color = centers[i]
                    brightness = int(mean_color[0]) + int(mean_color[1]) + int(mean_color[2])
                    cluster_list.append({
                        "color": mean_color,
                        "points": sort_points_snake(pts),
                        "brightness": brightness
                    })

        cluster_list.sort(key=lambda c: c["brightness"])

        total_outline_points = sum(len(path) for path in contour_paths)
        total_fill_points = sum(len(c["points"]) for c in cluster_list)

        speed_outline = 300  
        speed_fill = 1500  

        outline_frames = max(30, int(total_outline_points / speed_outline))
        fill_frames = max(30, int(total_fill_points / speed_fill)) if total_fill_points > 0 else 0
        exit_frames = 25

        total_frames = outline_frames + fill_frames + exit_frames

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        def overlay_image_fast(background, foreground, x, y):
            bg_h, bg_w = background.shape[:2]
            fg_h, fg_w = foreground.shape[:2]
            x_offset = x - int(fg_w * 0.15)
            y_offset = y - int(fg_h * 0.15)
            y1, y2 = max(0, y_offset), min(bg_h, y_offset + fg_h)
            x1, x2 = max(0, x_offset), min(bg_w, x_offset + fg_w)
            if y1 >= y2 or x1 >= x2:
                return background
            fy1, fy2 = y1 - y_offset, y1 - y_offset + (y2 - y1)
            fx1, fx2 = x1 - x_offset, x1 - x_offset + (x2 - x1)
            fg_roi = foreground[fy1:fy2, fx1:fx2]
            bg_roi = background[y1:y2, x1:x2]
            if fg_roi.shape[2] == 4:
                alpha = fg_roi[:, :, 3:4] / 255.0
                fg_rgb = fg_roi[:, :, :3]
                background[y1:y2, x1:x2] = (1.0 - alpha) * bg_roi + alpha * fg_rgb
            else:
                background[y1:y2, x1:x2] = fg_roi[:, :, :3]
            return background

        final_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        for cluster in cluster_list:
            pts = cluster["points"]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                final_canvas[ys, xs] = img[ys, xs]
        for path in contour_paths:
            if len(path) > 1:
                cv2.polylines(final_canvas, [np.array(path, np.int32)], isClosed=False, color=(0, 0, 0), thickness=2)

        last_hand_pos = (width // 2, height // 2)

        for frame_idx in range(total_frames):
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            current_hand_pos = last_hand_pos
            drawing_end_frame = total_frames - exit_frames

            if frame_idx < drawing_end_frame:
                if frame_idx < outline_frames:
                    progress = frame_idx / outline_frames
                    target_pts_count = int(total_outline_points * min(1.0, progress))
                    accumulated = 0
                    for path in contour_paths:
                        path_len = len(path)
                        if target_pts_count >= accumulated + path_len:
                            pts_arr = np.array(path, np.int32)
                            cv2.polylines(canvas, [pts_arr], isClosed=False, color=(0, 0, 0), thickness=2)
                            current_hand_pos = path[-1]
                            accumulated += path_len
                        elif target_pts_count > accumulated:
                            take_n = target_pts_count - accumulated
                            partial_path = path[:take_n]
                            if len(partial_path) > 1:
                                pts_arr = np.array(partial_path, np.int32)
                                cv2.polylines(canvas, [pts_arr], isClosed=False, color=(0, 0, 0), thickness=2)
                            if len(partial_path) > 0:
                                current_hand_pos = partial_path[-1]
                            break
                        else:
                            break
                    last_hand_pos = current_hand_pos
                else:
                    for path in contour_paths:
                        if len(path) > 1:
                            cv2.polylines(canvas, [np.array(path, np.int32)], isClosed=False, color=(0, 0, 0), thickness=2)

                    fill_idx = frame_idx - outline_frames
                    fill_progress = fill_idx / fill_frames if fill_frames > 0 else 1.0
                    target_accumulated_pts = int(total_fill_points * min(1.0, fill_progress))

                    current_accumulated = 0
                    for cluster in cluster_list:
                        pts = cluster["points"]
                        cluster_len = len(pts)
                        if target_accumulated_pts >= current_accumulated + cluster_len:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            canvas[ys, xs] = img[ys, xs]
                            current_hand_pos = pts[-1]
                            current_accumulated += cluster_len
                        elif target_accumulated_pts > current_accumulated:
                            take_count = target_accumulated_pts - current_accumulated
                            active_pts = pts[:take_count]
                            if len(active_pts) > 0:
                                xs = [p[0] for p in active_pts]
                                ys = [p[1] for p in active_pts]
                                canvas[ys, xs] = img[ys, xs]
                                current_hand_pos = active_pts[-1]
                            break
                        else:
                            break
                    last_hand_pos = current_hand_pos
            else:
                canvas = final_canvas.copy()
                exit_progress = (frame_idx - drawing_end_frame) / exit_frames
                exit_x = int(last_hand_pos[0] + exit_progress * (width - last_hand_pos[0] + 300))
                exit_y = int(last_hand_pos[1] + exit_progress * (-200 - last_hand_pos[1]))
                current_hand_pos = (exit_x, exit_y)

            if hand_img is not None:
                canvas = overlay_image_fast(canvas, hand_img, current_hand_pos[0], current_hand_pos[1])

            out.write(canvas)

        out.release()
        
        # Sukses render, catat waktu akses IP tersebut
        IP_COOLDOWN[client_ip] = current_time

    except Exception as e:
        return jsonify({"error": f"Gagal memproses video: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "message": "Video berhasil dirender!",
        "download_url": f"/download/{video_filename}"
    })

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
