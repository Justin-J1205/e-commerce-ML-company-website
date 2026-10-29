"""
app.py — Local Flask Web Server for JMI Commerce
Serves an executive light-mode website connected to models.py.
Exposes real-time endpoints for machine learning inference and training accuracy scores.
"""

from flask import Flask, render_template, request, jsonify
from models import JMICommerceModels

app = Flask(__name__)

# Initialize ML Engine using Retail_sales.csv
print("Initializing JMI Commerce ML Engine...")
ml_engine = JMICommerceModels(csv_path='Retail_sales.csv')
print("ML Engine loaded successfully!")
print("Model Accuracy Scores:", ml_engine.get_model_accuracies())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns dataset metrics and trained model accuracy percentages."""
    return jsonify({
        'total_records': f"{len(ml_engine.df):,}",
        'unique_products': f"{ml_engine.df['ITEM CODE'].nunique():,}",
        'suppliers': f"{ml_engine.df['SUPPLIER'].nunique():,}",
        'avg_sales': f"${ml_engine.df['RETAIL SALES'].mean():.2f}",
        'accuracies': ml_engine.get_model_accuracies()
    })

@app.route('/api/accuracies', methods=['GET'])
def get_accuracies():
    """Endpoint returning model training accuracy percentages."""
    return jsonify(ml_engine.get_model_accuracies())

@app.route('/api/predict_sales', methods=['POST'])
def predict_sales():
    data = request.get_json()
    sales = ml_engine.predict_sales(
        float(data.get('warehouse_sales', 0)),
        float(data.get('retail_transfers', 0)),
        int(data.get('month', 1))
    )
    return jsonify({
        'predicted_sales': round(sales, 2),
        'model_accuracy': f"{ml_engine.lr_accuracy}%"
    })

@app.route('/api/predict_risk', methods=['POST'])
def predict_risk():
    data = request.get_json()
    pred, prob = ml_engine.predict_slow_moving(
        float(data.get('retail_sales', 0)),
        float(data.get('warehouse_sales', 0)),
        float(data.get('retail_transfers', 0)),
        int(data.get('month', 1))
    )
    return jsonify({
        'is_slow_moving': pred,
        'probability': round(prob * 100, 1),
        'model_accuracy': f"{ml_engine.dt_accuracy}%"
    })

@app.route('/api/predict_anomaly', methods=['POST'])
def predict_anomaly():
    data = request.get_json()
    pred, prob = ml_engine.predict_anomaly(
        float(data.get('retail_sales', 0)),
        float(data.get('retail_transfers', 0)),
        float(data.get('warehouse_sales', 0))
    )
    return jsonify({
        'is_anomaly': pred,
        'probability': round(prob * 100, 1),
        'model_accuracy': f"{ml_engine.svm_accuracy}%"
    })

@app.route('/api/find_substitutes', methods=['POST'])
def find_substitutes():
    data = request.get_json()
    subs = ml_engine.find_substitutes(
        float(data.get('target_sales', 0)),
        float(data.get('target_transfers', 0)),
        float(data.get('target_warehouse', 0)),
        n_recommendations=3
    )
    return jsonify({
        'substitutes': subs,
        'model_accuracy': ml_engine.knn_accuracy
    })

@app.route('/api/classify_item', methods=['POST'])
def classify_item():
    data = request.get_json()
    desc = data.get('item_description', '')
    res = ml_engine.classify_item_type(desc)
    res['model_accuracy'] = f"{ml_engine.nb_accuracy}%"
    return jsonify(res)

@app.route('/api/predict_velocity', methods=['POST'])
def predict_velocity():
    data = request.get_json()
    pred, prob = ml_engine.predict_high_velocity(
        float(data.get('warehouse_sales', 0)),
        float(data.get('retail_transfers', 0)),
        int(data.get('month', 1))
    )
    return jsonify({
        'is_top_seller': pred,
        'probability': round(prob * 100, 1),
        'model_accuracy': f"{ml_engine.log_accuracy}%"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)