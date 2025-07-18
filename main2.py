from flask import Flask, request, jsonify
from model import rekomen  # Impor fungsi rekomen dari model.py

app = Flask(__name__)

@app.route('/rekomendasi', methods=['GET'])
def get_rekomendasi():
    """
    Endpoint GET untuk mendapatkan rekomendasi wisata.
    Query parameters:
        - var_update (int): 0 untuk data lama, 1 untuk kalkulasi baru (default: 0)
        - user (str): Nama user (default: "diana")
    Returns:
        JSON: Rekomendasi dalam format tabel
    """
    # Ambil parameter dari query string
    var_update = request.args.get('var_update', default=0, type=int)
    user = request.args.get('user', default="reviewer529", type=str)

    # Jalankan fungsi rekomen
    try:
        result_df = rekomen(var_update=var_update, user=user)
        if result_df is None:
            return jsonify({
                "status": "error",
                "message": "Fungsi rekomen mengembalikan None"
            }), 500

        # Konversi DataFrame ke format JSON (list of dictionaries)
        result_json = result_df.to_dict(orient="records")

        # Kembalikan respons JSON
        return jsonify({
            "status": "success",
            "user": user,
            "data": result_json
        }), 200
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)