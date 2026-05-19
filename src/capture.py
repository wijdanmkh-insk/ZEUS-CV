import cv2
import os
import glob
import re

def get_next_sequence_and_item(folder_path):
    """
    Tentukan nomor urut file berikutnya dengan mencari nomor terbesar
    dari file yang sudah ada (mis. R_0220.jpg) sehingga penomoran
    tidak akan mengulang walau ada file yang terhapus.
    Mengembalikan tuple (file_idx, item_num, shot_num).
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return 1, 1, 1

    pattern = re.compile(rf"{re.escape(os.path.basename(folder_path))}_(\d+)\.jpg$")
    files = glob.glob(os.path.join(folder_path, "*.jpg"))

    max_idx = 0
    for f in files:
        name = os.path.basename(f)
        m = pattern.match(name)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue

    # next index is max existing + 1 (or 1 if none)
    file_idx = max_idx + 1

    # Determine item_num and shot_num based on file_idx-1 (existing count by index)
    existing_count = max_idx
    item_num = (existing_count // 10) + 1
    shot_num = (existing_count % 10) + 1

    return file_idx, item_num, shot_num

def main():
    # Mapping folder sesuai request
    categories = {
        '1': {'name': 'Organik', 'folder': 'O'},
        '2': {'name': 'Anorganik', 'folder': 'R'},
        '3': {'name': 'Kertas', 'folder': 'P'},
        '4': {'name': 'Hazard', 'folder': 'H'}
    }

    print("=== DATA COLLECTOR SAMPAH V1 ===")
    print("Pilih Kategori Sampah yang akan diambil gambarnya:")
    for key, val in categories.items():
        print(f"[{key}] {val['name']} (Folder: {val['folder']}/)")
        
    choice = input("Masukkan pilihan (1/2/3/4): ").strip()
    
    if choice not in categories:
        print("Pilihan tidak valid. Program keluar.")
        return

    selected_folder = categories[choice]['folder']
    selected_name = categories[choice]['name']
    
    # Ambil index counter terakhir yang ada di folder agar tidak overwrite data lama
    file_idx, item_num, shot_num = get_next_sequence_and_item(selected_folder)

    # Inisialisasi Kamera Eksternal
    # Jika kamera laptop = 0, biasanya USB Cam/Kamera Eksternal = 1 atau 2.
    # Ubah ke 0 jika ingin mencoba menggunakan webcam bawaan laptop terlebih dahulu.
    camera_index = 1 
    cap = cv2.VideoCapture(camera_index)

    # Set resolusi ke 1080p atau 720p jika kamera mendukung (Opsional untuk Edge AI)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Gagal membuka kamera pada indeks {camera_index}. Mencoba kamera internal (indeks 0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Semua kamera gagal dibuka.")
            return

    print(f"\n[READY] Kategori: {selected_name} | Folder: {selected_folder}/")
    print("=== KONTROL ===")
    print("- Tekan [SPACE] untuk mengambil gambar")
    print("- Tekan [Q] atau [ESC] untuk keluar")
    print("-----------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal menerima frame dari kamera.")
            break

        # Clone frame asli untuk GUI overlay agar text tidak ikut tersimpan ke dataset
        display_frame = frame.copy()

        # Menampilkan status counter di layar preview (Osd)
        status_text_1 = f"Kategori: {selected_name} ({selected_folder}/)"
        status_text_2 = f"Item ke-{item_num} | Shot: {shot_num}/10"
        status_text_3 = f"Total Terpola: {file_idx - 1} Gambar"

        # Overlay text ke layar preview
        cv2.putText(display_frame, status_text_1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, status_text_2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, status_text_3, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Tampilkan window preview
        cv2.imshow("Dataset Collector Window", display_frame)

        # Logic Keyboard Event
        key = cv2.waitKey(1) & 0xFF
        
        # Jika tekan SPACE
        if key == 32:
            # Format nama file: folder/prefix_nomor.jpg (Misal: R/R_0023.jpg)
            # Menggunakan padding 4 digit agar penamaan rapi saat sorting di Linux/Windows
            filename = f"{selected_folder}/{selected_folder}_{file_idx:04d}.jpg"
            
            # Simpan frame asli (bukan display_frame yang ada text-nya)
            cv2.imwrite(filename, frame)
            print(f"[SAVED] {filename} -> (Item ke-{item_num}, Shot {shot_num}/10)")
            
            # Naikkan counter
            file_idx += 1
            shot_num += 1
            
            # Jika shot sudah mencapai urutan ke-11, reset shot ke 1 dan naikkan item_num
            if shot_num > 10:
                print(f"\n[INFO] Batch Item ke-{item_num} SELESAI. Silakan ganti objek ke-{item_num + 1}...")
                shot_num = 1
                item_num += 1
                print("-----------------------------------------------")

        # Jika tekan 'q' atau ESC
        elif key == ord('q') or key == 27:
            print("\nPengambilan dataset dihentikan.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()