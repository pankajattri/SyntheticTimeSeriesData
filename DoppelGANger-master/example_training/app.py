# app.py
import os
from flask import Flask, request, jsonify
from gan_task import GANTask
from config import config
from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler

app = Flask(__name__)

@app.route('/upload/<project_id>', methods=['POST'])
def upload_files(project_id):
    try:
        # Define the target folder dynamically
        target_folder = f'/app/work/{project_id}/train_data'
        
        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)
        
        # Check if files are in the request
        if 'files' not in request.files:
            return jsonify({"error": "No files provided in the request."}), 400
        
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({"error": "No files to upload."}), 400
        
        uploaded_files = []
        for file in files:
            if file.filename == '':
                continue
            # Save each file to the target folder
            file_path = os.path.join(target_folder, file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
        
        if not uploaded_files:
            return jsonify({"error": "No valid files uploaded."}), 400
        
        # Return success response with the list of uploaded files
        return jsonify({"message": "Files uploaded successfully.", "uploaded_files": uploaded_files}), 200
    
    except Exception as e:
        # Return error response if something goes wrong
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

@app.route('/download/<project_id>/<filename>')
def download_file(project_id, filename):
    # Construct the directory path dynamically using the project_id
    directory = f'/app/results/{project_id}'
    
    try:
        # Serve the requested file from the dynamically determined directory
        return send_from_directory(directory=directory, filename=filename, as_attachment=True)
    except FileNotFoundError:
        # Handle the case where the file or directory does not exist
        abort(404, description="File not found.")

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
        results = {"status": "success","message": f"GAN model executed successfully. Model checkpoints stored in /app/work/{params["projectid"]}/train_data"}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Expose Flask app to the network
