import os
import logging
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from openai import OpenAI
from pydub import AudioSegment
import tempfile
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, 
    template_folder='templates',  # specify the templates folder
    static_folder='static'        # specify the static folder
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
MAX_SEGMENT_SIZE = 25 * 1024 * 1024  # 25MB in bytes
SEGMENT_DURATION = 25 * 60 * 1000     # 25 minutes in milliseconds

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize OpenAI client
client = OpenAI()

# Add near other app configurations
progress_data = {}
progress_lock = Lock()

def setup_upload_folder():
    """Create the uploads directory if it doesn't exist."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logging.info(f"Upload folder '{UPLOAD_FOLDER}' is set up.")

@app.route('/')
def serve_index():
    logging.info("Serving index.html")
    return send_from_directory('templates', 'index.html')

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
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        logging.info(f"Transcribed segment: {file_path}")
        
        # Log the response for debugging
        logging.debug(f"Transcript response type: {type(transcript)}")
        logging.debug(f"Transcript response: {transcript}")
        
        # Extract just the text field from the response
        return transcript.text
        
    except Exception as e:
        logging.error(f"Error in transcribe_segment: {str(e)}")
        # Log the full error details
        logging.error(f"Full error details: {str(e.__dict__)}")
        raise

def cleanup_segments(segment_files):
    """Clean up temporary segment files"""
    for file_path in segment_files:
        try:
            os.remove(file_path)
            logging.info(f"Removed temporary segment file: {file_path}")
        except Exception as e:
            logging.error(f"Error cleaning up {file_path}: {str(e)}")

def update_progress(session_id, current, total):
    """Update progress for a given session"""
    with progress_lock:
        progress_data[session_id] = {
            'current': current,
            'total': total,
            'percentage': (current / total * 100) if total > 0 else 0,
            'status': 'processing' if current < total else 'complete'
        }

def process_audio_file(file_path, session_id):
    """Process audio file with progress tracking"""
    file_size = os.path.getsize(file_path)
    
    # Initialize progress before starting processing
    update_progress(session_id, 0, 1)  # Default to 1 segment
    
    if file_size > MAX_SEGMENT_SIZE:
        segments = split_audio(file_path)
        transcriptions = []
        total_segments = len(segments)
        
        # Update progress with actual total segments
        update_progress(session_id, 0, total_segments)
        
        try:
            for index, segment in enumerate(segments, 1):
                segment_transcription = transcribe_segment(segment)
                transcriptions.append(segment_transcription)
                update_progress(session_id, index, total_segments)
        finally:
            cleanup_segments(segments)
            
        return ' '.join(transcriptions)
    else:
        result = transcribe_segment(file_path)
        update_progress(session_id, 1, 1)  # Mark as complete
        return result

@app.route('/check-progress/<session_id>')
def check_progress(session_id):
    """Endpoint for checking transcription progress"""
    with progress_lock:
        if session_id not in progress_data:
            return jsonify({'error': 'Session not found'}), 404
        return jsonify(progress_data[session_id])

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        if 'audio' not in request.files:
            logging.error("No file provided in the request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['audio']
        logging.info(f"Received file: {file.filename}")
        
        if file.filename == '':
            logging.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logging.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Log before saving
        logging.info(f"Attempting to save file to: {file_path}")
        file.save(file_path)
        logging.info(f"Successfully saved file to: {file_path}")
        
        # Log before transcription
        logging.info("Starting transcription process")
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
        # Initialize progress data immediately
        update_progress(session_id, 0, 1)  # Start with default values
        
        # Process file in a separate thread to not block progress checks
        def process_async():
            try:
                transcription = process_audio_file(file_path, session_id)
                
                # Create blog post and update response
                blog_post = {
                    'title': f"Transcript: {filename}",
                    'keyword': 'audio_transcript',
                    'content': transcription
                }
                
                if not hasattr(app, 'blog_posts'):
                    app.blog_posts = []
                
                app.blog_posts.append(blog_post)
                post_id = len(app.blog_posts) - 1
                
                # Update progress data with success
                with progress_lock:
                    progress_data[session_id].update({
                        'status': 'complete',
                        'success': True,
                        'id': post_id
                    })
                    
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                with progress_lock:
                    progress_data[session_id].update({
                        'status': 'error',
                        'error': str(e)
                    })
        
        # Start processing in background
        from threading import Thread
        Thread(target=process_async).start()
        
        # Return immediate response
        return jsonify({
            'success': True,
            'message': 'Processing started',
            'session_id': session_id
        })
        
    except Exception as e:
        logging.error(f"Unexpected error in upload_audio: {str(e)}")
        logging.exception("Full traceback:")
        return jsonify({'error': 'An unexpected error occurred. Please check the logs.'}), 500

@app.route('/blog')
def view_blog():
    post_id = request.args.get('id', type=int)
    if not hasattr(app, 'blog_posts') or post_id is None or post_id >= len(app.blog_posts):
        return "Blog post not found", 404
    
    try:
        # Get example blog text for the rewrite feature
        example_blog_path = os.path.join(app.static_folder, 'example_blog.txt')
        with open(example_blog_path, 'r') as f:
            blog_txt = f.read()
        
        return render_template('blog.html', 
                             blog_posts=[app.blog_posts[post_id]], 
                             blog_txt=blog_txt)
    except Exception as e:
        logging.error(f"Error rendering blog template: {str(e)}")
        return f"Error rendering blog template: {str(e)}", 500

if __name__ == '__main__':
    setup_upload_folder()  # Ensure the uploads directory exists
    app.run(debug=True)
