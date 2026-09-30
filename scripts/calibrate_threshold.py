"""Kalibrasi retrieval_threshold: jalankan search_similar() untuk query di
dalam cakupan FAQ vs di luar cakupan, lalu bandingkan skor top-1 tiap
kelompok untuk mengusulkan nilai threshold yang memisahkan keduanya.

Jalankan: PYTHONPATH=. python scripts/calibrate_threshold.py
"""

from dotenv import load_dotenv

load_dotenv()

from app.services.rag import search_similar  # noqa: E402

IN_SCOPE_QUERIES = [
    "berapa lama garansi headset",
    "apa bisa COD",
    "cara retur barang",
    "berapa hari pengiriman ke luar jabodetabek",
    "metode pembayaran apa saja yang diterima",
    "bagaimana cara melacak pesanan saya",
]

OUT_OF_SCOPE_QUERIES = [
    "apa ada diskon member",
    "cara ganti password",
    "rekomendasi laptop",
    "cuaca hari ini",
    "siapa presiden indonesia",
    "resep nasi goreng",
]


def _top_score(query: str) -> float:
    results = search_similar(query, top_k=1)
    return results[0]["score"] if results else 0.0


def main() -> None:
    print("=" * 70)
    print("QUERY DALAM CAKUPAN FAQ")
    print("=" * 70)
    in_scope_scores = []
    for q in IN_SCOPE_QUERIES:
        score = _top_score(q)
        in_scope_scores.append(score)
        print(f"  {score:.4f}  \"{q}\"")

    print()
    print("=" * 70)
    print("QUERY DI LUAR CAKUPAN FAQ")
    print("=" * 70)
    out_of_scope_scores = []
    for q in OUT_OF_SCOPE_QUERIES:
        score = _top_score(q)
        out_of_scope_scores.append(score)
        print(f"  {score:.4f}  \"{q}\"")

    min_in_scope = min(in_scope_scores)
    max_out_of_scope = max(out_of_scope_scores)

    print()
    print("=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    print(f"Skor top-1 dalam cakupan   -> min={min_in_scope:.4f}  max={max(in_scope_scores):.4f}")
    print(f"Skor top-1 di luar cakupan -> min={min(out_of_scope_scores):.4f}  max={max_out_of_scope:.4f}")

    if min_in_scope > max_out_of_scope:
        suggested = (min_in_scope + max_out_of_scope) / 2
        print(f"\nDua kelompok terpisah bersih (gap={min_in_scope - max_out_of_scope:.4f}).")
        print(f"Usulan retrieval_threshold (titik tengah): {suggested:.4f}")
    else:
        suggested = (min_in_scope + max_out_of_scope) / 2
        print(
            "\nPERINGATAN: dua kelompok TUMPANG TINDIH "
            f"(skor in-scope terendah {min_in_scope:.4f} <= skor out-of-scope tertinggi {max_out_of_scope:.4f})."
        )
        print(f"Titik tengah tetap dihitung sebagai estimasi kasar: {suggested:.4f}")
        print("Threshold ini tidak akan memisahkan kedua kelompok dengan sempurna, perlu query kalibrasi lebih banyak.")

    print(f"\n=> Simpan sebagai RETRIEVAL_THRESHOLD = {suggested:.2f} di app/services/rag.py")


if __name__ == "__main__":
    main()
