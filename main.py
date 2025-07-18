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
        # result_df = rekomen(var_update=var_update, user=user)
        # if result_df is None:
        #     return jsonify({
        #         "status": "error",
        #         "message": "Fungsi rekomen mengembalikan None"
        #     }), 500

        # # Konversi DataFrame ke format JSON (list of dictionaries)
        # result_json = result_df.to_dict(orient="records")

        # Kembalikan respons JSON
        return jsonify({
                    "data": [
                        {
                        "Predicted_Item": 0.4932598453340854,
                        "Predicted_User": 0.7400076013001999,
                        "Weighted_Average": 0.690658050106977,
                        "nama DTW": "klenteng hok tiek bio"
                        },
                        {
                        "Predicted_Item": 0.5422492451559303,
                        "Predicted_User": 0.6482399571230639,
                        "Weighted_Average": 0.6270418147296372,
                        "nama DTW": "taman lalu lintas anak bangsa"
                        },
                        {
                        "Predicted_Item": 0.5422712999594942,
                        "Predicted_User": 0.6439888637481465,
                        "Weighted_Average": 0.6236453509904161,
                        "nama DTW": "cagar alam geologi karangsambung"
                        },
                        {
                        "Predicted_Item": 0.5643803366932896,
                        "Predicted_User": 0.6362342294794888,
                        "Weighted_Average": 0.621863450922249,
                        "nama DTW": "agro wisata amanah"
                        },
                        {
                        "Predicted_Item": 0.4932059091671349,
                        "Predicted_User": 0.6364710894525589,
                        "Weighted_Average": 0.6078180533954741,
                        "nama DTW": "sanggaluri park"
                        },
                        {
                        "Predicted_Item": 0.4931801797396457,
                        "Predicted_User": 0.6352038275859612,
                        "Weighted_Average": 0.6067990980166981,
                        "nama DTW": "saloka theme park"
                        },
                        {
                        "Predicted_Item": 0.514836549043292,
                        "Predicted_User": 0.6230242245764545,
                        "Weighted_Average": 0.601386689469822,
                        "nama DTW": "museum ohd"
                        },
                        {
                        "Predicted_Item": 0.4931859535653527,
                        "Predicted_User": 0.6090469043801581,
                        "Weighted_Average": 0.5858747142171971,
                        "nama DTW": "pulau panjang jepara"
                        },
                        {
                        "Predicted_Item": 0.5435178043275719,
                        "Predicted_User": 0.5863560863620266,
                        "Weighted_Average": 0.5777884299551357,
                        "nama DTW": "bukit tranggulasih"
                        },
                        {
                        "Predicted_Item": 0.4911577232331938,
                        "Predicted_User": 0.5977696741584838,
                        "Weighted_Average": 0.5764472839734258,
                        "nama DTW": "taman diponegoro"
                        }
                    ],
                    "status": "success",
                    "user": "reviewer529"
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