# app.py
import os
import time, zipfile, io, json
from flask import Flask, request, jsonify, Response, send_file, copy_current_request_context
from gan_task import GANTask
from gan_generate_data_task import GANGenerateDataTask
from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler

app = Flask(__name__)


def upload_files(project_id,files):
    try:
        # Define the target folder dynamically
        target_folder = '/app/work/{}/train_data'.format(project_id)
        
        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)
        
                
        uploaded_files = []
        for file in files:
            if file.filename == '':
                continue
            print('File ',file.filename)
            # Save each file to the target folder
            file_path = os.path.join(target_folder, file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
        
        # Return success response with the list of uploaded files
        return json.dumps({"message": "Files uploaded successfully.", "status": 200, "uploaded_files": uploaded_files})
    
    except Exception as e:
        # Return error response if something goes wrong
        return json.dumps({"error": "File upload failed: {}".format(str(e)), "status": 500})
    
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

def run_model(params):
    """
    API endpoint to run the GAN model.
    Expects a JSON payload with parameters.
    """
    time.sleep(2)
    try:
        # Get parameters from the HTTP request
        #params = request.get_json()
        '''
        params = request.form.get('params')  # if sent as a string
        if params:
            params = json.loads(params)
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        '''
        # Run the GAN model with the parameters
        scheduler = GPUTaskScheduler(config=params, gpu_task_class=GANTask)
        scheduler.start()
        results = {"status": "success","message": "GAN model trained successfully."}
        #return jsonify(results), 200
        return results
    except Exception as e:
        #return jsonify({"error": str(e)}), 500
        return {"status": "GAN model train step failed. ","message": str(e)}
    

def generate_data(params):
    """
    API endpoint to generate synthetic data using the GAN model.
    Expects a JSON payload with parameters.
    """
    time.sleep(2)
    try:
        # Get parameters from the HTTP request
        '''
        params = request.get_json()
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        '''
        # Run the GAN model with the parameters
        scheduler = GPUTaskScheduler(config=params, gpu_task_class=GANGenerateDataTask)
        scheduler.start()
        results = {"status": "success","message": "Synthetic Data generated successfully."}
        #return jsonify(results), 200
        return results
    except Exception as e:
        #return jsonify({"error": str(e)}), 500
        return {"status": "Generate data step failed. ","message":str(e)}

@app.route('/full-workflow1/<project_id>', methods=['POST'])
def full_workflow1(project_id):
    # Extract files upfront before entering the generator
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided in the request."}), 400
        file_details = [{"filename": file.filename} for file in files]
    except Exception as e:
        return jsonify({"error": "File extraction failed: {}".format(str(e))}), 500

    # Generator for the workflow
    def workflow_generator(file_details):
        # Step 1: Handle file details
        yield jsonify({"status": "uploading files", "step": "upload", "details": file_details}), 200

        # Step 2: Example placeholder for additional steps
        yield jsonify({"status": "completed", "step": "next_step"}), 200

    return Response(workflow_generator(file_details), content_type="application/json")



'''
@app.route('/full-workflow/<project_id>', methods=['POST'])
def full_workflow(project_id):
    files = request.files.getlist('files')
    return jsonify({"status": "uploading files", "step": "upload", "details": files[0].filename}), 200
'''    

@app.route('/upload-train/<project_id>', methods=['POST'])
def upload_train(project_id):
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files provided in the request."}), 400
        
        # Define the target folder dynamically
        target_folder = '/app/work/{}/train_data'.format(project_id)
        
        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)
        uploaded_files = []
        for file in files:
            if file.filename:
                file_path = os.path.join(target_folder, file.filename)
                file.save(file_path)
                uploaded_files.append(file.filename)

        params = request.form.get('params')  # if sent as a string
        if params:
            params = json.loads(params)
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        
    except Exception as e:
        return jsonify({"error": "Initialization  failed: {}".format(str(e))}), 500

    def workflow_generator():
        try:
            
            # Step 1: Confirm file upload
            yield json.dumps({
                "message": "Files uploaded successfully and params received.",
                "status": 200,
                "uploaded_files": uploaded_files
            }) + '\n'
            '''
            upload_response = upload_files(project_id,files)
            
            if 200 != 200:
                yield json.dumps({"status": "failed", "step": "upload", "details": upload_response.json()}) + '\n'
                
            else:
                yield json.dumps({"status": "completed", "step": "upload", "details": upload_response.json()}) + '\n'
            '''
            
            # Step 2: Start training
            training_response = run_model(params)
            yield json.dumps(training_response) + '\n'
            '''
            if training_response.status_code != 200:
                yield json.dumps({"status": "failed", "step": "training", "details": training_response.json()}) + '\n'
                
            else:
                yield json.dumps({"status": "completed", "step": "training", "details": training_response.json()}) + '\n'
            '''
            
            # Step 3: Generate synthetic data
            #generate_response = generate_data(params)
            #yield json.dumps(generate_response) + '\n'
            '''
            if generate_response.status_code != 200:
                yield json.dumps({"status": "failed", "step": "generation", "details": generate_response.json()}) + '\n'
                
            else:
                yield json.dumps({"status": "completed", "step": "generation", "details": generate_response.json()}) + '\n'
            '''
            
        except Exception as e:
            yield json.dumps({"error": "An unexpected error occurred: {}".format(str(e))})

    return Response(workflow_generator(), content_type="application/json")

@app.route('/generate/<project_id>', methods=['POST'])
def generate(project_id):
    try:
        params = request.form.get('params')  # if sent as a string
        if params:
            params = json.loads(params)
        if not params:
            return jsonify({"error": "No parameters provided"}), 400
        
    except Exception as e:
        return jsonify({"error": "Initialization  failed: {}".format(str(e))}), 500

    def workflow_generator():
        try:
            
            # Step 3: Generate synthetic data
            generate_response = generate_data(params)
            yield json.dumps(generate_response) + '\n'
            
        except Exception as e:
            yield json.dumps({"error": "An unexpected error occurred: {}".format(str(e))})

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
                    if file.endswith("_generated_data_train.npz"):
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
            attachment_filename='synthetic_data_{}.zip'.format(project_id)
        )
    except Exception as e:
        return jsonify({"error": "Failed to download files: {}".format(str(e))}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Expose Flask app to the network
