import cv2
from ultralytics import YOLO

# 1. Muat model .pt Anda yang sudah terlatih
# Pastikan path ke file .pt sudah benar
try:
    model = YOLO("best.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Inisialisasi penangkapan video dari webcam (kamera no. 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# 3. Loop untuk memproses setiap frame dari video secara real-time
while True:
    # Baca satu frame dari kamera
    success, frame = cap.read()

    if success:
        # 4. Lakukan deteksi pada frame
        # 'stream=True' lebih efisien untuk video
        results = model(frame, stream=True, conf=0.6)

        # 5. Proses hasil deteksi dan gambar di atas frame
        for r in results:
            # .plot() adalah fungsi bawaan ultralytics yang sangat nyaman
            # untuk langsung menggambar kotak dan label pada frame
            annotated_frame = r.plot()

            # 6. Tampilkan frame yang sudah di-anotasi
            cv2.imshow("Deteksi Alfabet SIBI Real-time", annotated_frame)

        # 7. Hentikan loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Jika gagal membaca frame, keluar dari loop
        break

# 8. Lepaskan sumber kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
