# app.py

from flask import Flask, request, jsonify
from gan_task import GANTask
from config import config
from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler

app = Flask(__name__)

@app.route("/run-model", methods=["POST"])
def run_model():
    """
    API endpoint to run the GAN model.
    Expects a JSON payload with parameters.
    """
    try:
        # Get parameters from the HTTP request
        params = request.get_json()
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        
        # Run the GAN model with the parameters
        scheduler = GPUTaskScheduler(config=params, gpu_task_class=GANTask)
        scheduler.start()
        results = {"status": "success","learning_rate": 0.0002,"epochs": 150,"message": "GAN model executed successfully."}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Expose Flask app to the network
