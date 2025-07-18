import argparse

parser = argparse.ArgumentParser(description="Perintah eksekusi program")
parser.add_argument("--nama", type=str, default="kosong", help="Nama pengguna")
parser.add_argument("--usia", type=int, help="Usia pengguna", default=0)

args = parser.parse_args()

print(f"Halo, {args.nama}! Usia kamu {args.usia} tahun.")