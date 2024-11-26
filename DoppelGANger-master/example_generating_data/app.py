# app.py
import os
from flask import Flask, request, jsonify
from gan_generate_data_task import GANGenerateDataTask
from config_generate_data import config
from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler

app = Flask(__name__)

@app.route('/download/<project_id>/<filename>')
def download_file(project_id, filename):
    # Construct the directory path dynamically using the project_id
    directory = '/app/work/{}/generated_data'.format(project_id)
    
    try:
        # Serve the requested file from the dynamically determined directory
        return send_from_directory(directory=directory, filename=filename, as_attachment=True)
    except FileNotFoundError:
        # Handle the case where the file or directory does not exist
        abort(404, description="File not found.")

@app.route("/generate-data", methods=["POST"])
def generate_data():
    """
    API endpoint to generate synthetic data using the GAN model.
    Expects a JSON payload with parameters.
    """
    try:
        # Get parameters from the HTTP request
        params = request.get_json()
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        
        # Run the GAN model with the parameters
        scheduler = GPUTaskScheduler(config=params, gpu_task_class=GANGenerateDataTask)
        scheduler.start()
        results = {"status": "success","message": "Synthetic Data generated successfully."}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Expose Flask app to the network
