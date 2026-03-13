"""Flask web frontend for Agentic Explainer.
Receives uploads and routes them to the LangGraph workflow.
"""
import os
import json
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

import config
from src.workflow import app_router

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.DATA_DIR
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_file():
    if 'document' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        
        enable_web_search = request.form.get('web_search') in ['true', 'True', 'on', '1']
        output_format = request.form.get('output_format', 'text')
        conf_threshold_raw = request.form.get('conf_threshold', str(config.VISION_CONF_THRESHOLD))
        anchor_priors_raw = request.form.get('anchor_priors', '')

        try:
            conf_threshold = float(conf_threshold_raw)
        except ValueError:
            return jsonify({"error": "Invalid conf_threshold. Must be a number between 0 and 1."}), 400

        if conf_threshold < 0 or conf_threshold > 1:
            return jsonify({"error": "Invalid conf_threshold. Must be between 0 and 1."}), 400

        anchor_priors = config.VISION_ANCHOR_PRIORS
        if anchor_priors_raw:
            try:
                parsed = json.loads(anchor_priors_raw)
                if not isinstance(parsed, list):
                    raise ValueError("anchor_priors must be a list")
                anchor_priors = parsed
            except Exception:
                return jsonify({"error": "Invalid anchor_priors. Expected JSON like [[0.2,0.2],[0.35,0.25]]."}), 400
        
        initial_state = {
            "file_path": file_path,
            "raw_text": "",
            "extracted_images": [],
            "vision_insights": [],
            "vision_output_images": [],
            "text_insights": "",
            "final_report": "",
            "needs_web_search": enable_web_search,
            "conf_threshold": conf_threshold,
            "anchor_priors": anchor_priors
        }
        
        try:
            final_state = app_router.invoke(initial_state)

            audio_path = None
            if output_format == 'audio':
                try:
                    from src.tools.tts_module import generate_audio_report
                    audio_path = generate_audio_report(final_state.get("final_report", ""))
                except Exception as e:
                    print(f"TTS generation failed: {e}")

            return jsonify({
                "message": "Processing complete",
                "report": final_state.get("final_report"),
                "format": output_format,
                "audio_file": audio_path,
                "vision_insights": final_state.get("vision_insights", []),
                "detected_images": final_state.get("vision_output_images", []),
                "conf_threshold": conf_threshold,
                "anchor_priors": anchor_priors
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    allowed_types = ", ".join(sorted(ext.upper() for ext in config.ALLOWED_EXTENSIONS))
    return jsonify({"error": f"Invalid file type. Allowed: {allowed_types}"}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
