# app.py
import os
import time, zipfile, io
from flask import Flask, request, jsonify, Response, send_file
from gan_generate_data_task import GANGenerateDataTask
from config_generate_data import config
from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler

app = Flask(__name__)


def upload_files(project_id):
    try:
        # Define the target folder dynamically
        target_folder = '/app/work/{}/train_data'.format(project_id)
        
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
        return jsonify({"error": "File upload failed: {}".format(str(e))}), 500
    
'''
def download_file(project_id, filename):
    # Construct the directory path dynamically using the project_id
    directory = '/app/work/{}/generated_data'.format(project_id)
    
    try:
        # Serve the requested file from the dynamically determined directory
        return send_from_directory(directory=directory, filename=filename, as_attachment=True)
    except FileNotFoundError:
        # Handle the case where the file or directory does not exist
        abort(404, description="File not found.")
'''

def run_model():
    """
    API endpoint to run the GAN model.
    Expects a JSON payload with parameters.
    """
    time.sleep(2)
    try:
        # Get parameters from the HTTP request
        params = request.get_json()
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        
        # Run the GAN model with the parameters
        scheduler = GPUTaskScheduler(config=params, gpu_task_class=GANTask)
        scheduler.start()
        results = {"status": "success","message": "GAN model trained successfully."}
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

def generate_data():
    """
    API endpoint to generate synthetic data using the GAN model.
    Expects a JSON payload with parameters.
    """
    time.sleep(2)
    try:
        # Get parameters from the HTTP request
        params = request.get_json()
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        
        # Run the GAN model with the parameters
        scheduler = GPUTaskScheduler(config=params, gpu_task_class=GANGenerateDataTask)
        scheduler.start()
        results = {"status": "success","message": "Synthetic Data generated successfully."}
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/full-workflow/<project_id>', methods=['POST'])
def full_workflow(project_id):
    def workflow_generator():
        try:
            # Step 1: Upload files
            upload_response = upload_files(project_id)
            if upload_response.status_code != 200:
                yield jsonify({"status": "failed", "step": "upload", "details": upload_response.json()}), 400
                return
            yield jsonify({"status": "completed", "step": "upload", "details": upload_response.json()})

            # Step 2: Start training
            training_response = run_model()
            if training_response.status_code != 200:
                yield jsonify({"status": "failed", "step": "training", "details": training_response.json()}), 400
                return
            yield jsonify({"status": "completed", "step": "training", "details": training_response.json()})

            # Step 3: Generate synthetic data
            generate_response = generate_data()
            if generate_response.status_code != 200:
                yield jsonify({"status": "failed", "step": "generation", "details": generate_response.json()}), 400
                return
            yield jsonify({"status": "completed", "step": "generation", "details": generate_response.json()})

            
        except Exception as e:
            yield jsonify({"error": "An unexpected error occurred: {}".format(str(e))}), 500

    return Response(workflow_generator(), content_type="application/json")

@app.route('/download/<project_id>', methods=['GET'])
def download_files(project_id):
    try:
        # Path to the synthetic data directory
        synthetic_data_path = '/app/work/{}/generated_data'.format(project_id)
        
        # Check if the directory exists
        if not os.path.exists(synthetic_data_path):
            return jsonify({"error": "No synthetic data found for project ID {}".format(project_id)}), 404
        
        # Create an in-memory ZIP file
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for root, dirs, files in os.walk(synthetic_data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, synthetic_data_path)  # Preserve directory structure
                    zf.write(file_path, arcname)
        
        # Seek to the beginning of the memory file for serving
        memory_file.seek(0)

        # Serve the ZIP file
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='synthetic_data_{}.zip'.format(project_id)
        )
    except Exception as e:
        return jsonify({"error": "Failed to download files: {}".format(str(e))}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Expose Flask app to the network
