import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
from pydub import AudioSegment
import tempfile

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
MAX_SEGMENT_SIZE = 25 * 1024 * 1024  # 25MB in bytes
SEGMENT_DURATION = 25 * 60 * 1000     # 25 minutes in milliseconds

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize OpenAI client
client = OpenAI()

def setup_upload_folder():
    """Create the uploads directory if it doesn't exist."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logging.info(f"Upload folder '{UPLOAD_FOLDER}' is set up.")

@app.route('/')
def serve_index():
    logging.info("Serving index.html")
    return send_from_directory('static', 'index.html')

def allowed_file(filename):
    is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    logging.debug(f"File '{filename}' allowed: {is_allowed}")
    return is_allowed

def calculate_optimal_chunk_duration(audio):
    """Calculate optimal chunk duration based on audio's bytes-per-second rate"""
    bytes_per_sec = (audio.frame_rate * audio.frame_width * audio.channels) / 8
    target_size = 24.5 * 1024 * 1024  # 24.5MB in bytes
    optimal_duration = (target_size / bytes_per_sec) * 1000  # Convert to milliseconds
    logging.debug(f"Calculated optimal chunk duration: {optimal_duration} ms")
    return int(optimal_duration)

def binary_search_optimal_duration(audio, start_duration, max_attempts=5):
    """Fine-tune chunk duration using binary search"""
    min_duration = start_duration // 2
    max_duration = start_duration * 1.5
    optimal_duration = start_duration
    
    for attempt in range(max_attempts):
        test_segment = audio[:optimal_duration]
        with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_file:
            test_segment.export(temp_file.name, format='mp3', parameters=["-b:a", "192k"])
            size = os.path.getsize(temp_file.name)
            logging.debug(f"Attempt {attempt + 1}: Test segment size: {size} bytes")
            
            if size > MAX_SEGMENT_SIZE:
                max_duration = optimal_duration
                optimal_duration = (min_duration + optimal_duration) // 2
                logging.info(f"Size too large, adjusting max_duration to {max_duration} ms")
            elif size < MAX_SEGMENT_SIZE * 0.9:  # Too small, try larger
                min_duration = optimal_duration
                optimal_duration = (optimal_duration + max_duration) // 2
                logging.info(f"Size too small, adjusting min_duration to {min_duration} ms")
            else:  # Good size (between 90% and 100% of target)
                logging.info(f"Optimal duration found: {optimal_duration} ms")
                break
    
    return optimal_duration

def split_audio(file_path):
    """Split audio file into optimally sized chunks"""
    try:
        audio = AudioSegment.from_file(file_path)
        logging.info(f"Loaded audio file: {file_path}")
    except Exception as e:
        logging.error(f"Failed to load audio file: {str(e)}")
        raise ValueError(f"Failed to load audio file: {str(e)}")
    
    if len(audio) == 0:
        logging.warning("Audio file appears to be empty")
        raise ValueError("Audio file appears to be empty")
    
    segments = []
    optimal_duration = calculate_optimal_chunk_duration(audio)
    
    try:
        optimal_duration = int(binary_search_optimal_duration(audio, optimal_duration))  # Convert to integer
        logging.info(f"Optimal chunk duration: {optimal_duration/1000:.2f} seconds")
        
        # Split the audio into segments
        for i in range(0, len(audio), optimal_duration):
            segment = audio[i:i + optimal_duration]
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                try:
                    segment.export(
                        temp_file.name,
                        format='mp3',
                        parameters=["-b:a", "192k"]
                    )
                    segments.append(temp_file.name)
                except Exception as e:
                    logging.error(f"Error exporting segment: {str(e)}")
                    # Clean up any temporary files
                    for seg_file in segments:
                        try:
                            os.remove(seg_file)
                        except:
                            pass
                    raise ValueError(f"Failed to export segment: {str(e)}")
        
        return segments
        
    except Exception as e:
        logging.warning(f"Error during audio splitting: {str(e)}")
        # Clean up any temporary files
        for seg_file in segments:
            try:
                os.remove(seg_file)
            except:
                pass
        raise ValueError(f"Error during audio splitting: {str(e)}")

def transcribe_segment(file_path):
    """Transcribe a single audio segment"""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    logging.info(f"Transcribed segment: {file_path}")
    
    # Log the entire response structure
    logging.debug(f"Full transcript response: {transcript}")
    if hasattr(transcript, 'words') and transcript.words:
        logging.debug(f"First word object: {vars(transcript.words[0])}")
    
    return []  # temporary return empty list until we see the structure

def cleanup_segments(segment_files):
    """Clean up temporary segment files"""
    for file_path in segment_files:
        try:
            os.remove(file_path)
            logging.info(f"Removed temporary segment file: {file_path}")
        except Exception as e:
            logging.error(f"Error cleaning up {file_path}: {str(e)}")

def process_audio_file(file_path):
    """Process audio file, splitting if necessary and transcribing"""
    file_size = os.path.getsize(file_path)
    logging.info(f"Processing audio file: {file_path}, size: {file_size} bytes")
    
    if file_size > MAX_SEGMENT_SIZE:
        segments = split_audio(file_path)
        transcriptions = []
        
        try:
            for segment in segments:
                segment_transcription = transcribe_segment(segment)
                transcriptions.extend(segment_transcription)
        finally:
            cleanup_segments(segments)
            
        return transcriptions
    else:
        return transcribe_segment(file_path)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        logging.error("No file provided in the request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logging.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(file_path)
        logging.info(f"Saved uploaded file: {file_path}")
        
        # Process and transcribe the file
        transcription = process_audio_file(file_path)
        
        return jsonify({
            'message': 'File processed successfully',
            'id': filename,
            'transcript': transcription
        })
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    setup_upload_folder()  # Ensure the uploads directory exists
    app.run(debug=True)
