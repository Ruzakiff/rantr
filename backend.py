import os
import logging
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from openai import OpenAI
from pydub import AudioSegment
import tempfile
from threading import Lock
import subprocess
import ffmpeg

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, 
    template_folder='templates',  # specify the templates folder
    static_folder='static'        # specify the static folder
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
MAX_SEGMENT_SIZE = 24 * 1024 * 1024  # 24MB to be safe
OPTIMAL_DURATION = 180  # Start with 3 minutes (usually ~4-5MB at 192kbps)
FALLBACK_DURATION = 90  # Fallback to 90 seconds if audio is dense

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
    """Fine-tune chunk duration using binary search with sample"""
    # Only test with first 30 seconds of audio for faster calculation
    test_audio = audio[:30000]  # First 30 seconds
    
    min_duration = int(start_duration // 2)  # Ensure integer
    max_duration = int(start_duration * 1.5)  # Ensure integer
    optimal_duration = int(start_duration)  # Ensure integer
    
    for attempt in range(max_attempts):
        test_segment = test_audio[:optimal_duration]
        with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_file:
            test_segment.export(temp_file.name, format='mp3', parameters=["-b:a", "192k"])
            size = os.path.getsize(temp_file.name)
            
            # Scale size estimate to full duration
            estimated_full_size = size * (len(audio) / len(test_audio))
            
            if estimated_full_size > MAX_SEGMENT_SIZE:
                max_duration = optimal_duration
                optimal_duration = int((min_duration + optimal_duration) // 2)  # Ensure integer
            elif estimated_full_size < MAX_SEGMENT_SIZE * 0.9:
                min_duration = optimal_duration
                optimal_duration = int((optimal_duration + max_duration) // 2)  # Ensure integer
            else:
                break
    
    return optimal_duration

def estimate_segments(file_size):
    """Estimate number of segments based on file size"""
    # Assume roughly 1MB per minute at 192kbps
    # Use 20MB as safe segment size for estimation
    estimated_segments = max(1, file_size // (20 * 1024 * 1024))
    return int(estimated_segments)

def split_audio_streaming(file_path):
    """Split audio file into chunks, optimizing for fewer segments while staying safe"""
    try:
        # Get duration using ffprobe
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        
        # Test first chunk to determine if we need smaller segments
        with tempfile.NamedTemporaryFile(suffix='.mp3') as test_file:
            stream = ffmpeg.input(file_path, ss=0, t=OPTIMAL_DURATION)
            stream = ffmpeg.output(stream, test_file.name, 
                acodec='libmp3lame', 
                b='192k',
                loglevel='error'
            )
            ffmpeg.run(stream, overwrite_output=True)
            
            # Determine if we need to use smaller chunks
            use_smaller_chunks = os.path.getsize(test_file.name) >= MAX_SEGMENT_SIZE
        
        # Set chunk size based on test
        chunk_duration = FALLBACK_DURATION if use_smaller_chunks else OPTIMAL_DURATION
        logging.info(f"Using {chunk_duration}s chunks based on initial test")
        
        # Process chunks
        current_time = 0
        while current_time < duration:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                try:
                    stream = ffmpeg.input(file_path, ss=current_time, t=chunk_duration)
                    stream = ffmpeg.output(stream, temp_file.name, 
                        acodec='libmp3lame', 
                        b='192k',
                        loglevel='error'
                    )
                    ffmpeg.run(stream, overwrite_output=True)
                    
                    # Final size check
                    if os.path.getsize(temp_file.name) >= MAX_SEGMENT_SIZE:
                        os.unlink(temp_file.name)
                        # Split into smaller chunks
                        for subchunk in range(2):
                            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as sub_file:
                                substream = ffmpeg.input(file_path, 
                                    ss=current_time + (subchunk * (chunk_duration/2)), 
                                    t=chunk_duration/2
                                )
                                substream = ffmpeg.output(substream, sub_file.name,
                                    acodec='libmp3lame',
                                    b='192k',
                                    loglevel='error'
                                )
                                ffmpeg.run(substream, overwrite_output=True)
                                yield sub_file.name
                    else:
                        yield temp_file.name
                
                except ffmpeg.Error as e:
                    logging.error(f"FFmpeg error: {e.stderr.decode()}")
                    raise
            
            current_time += chunk_duration
            
    except Exception as e:
        logging.error(f"Error in split_audio_streaming: {str(e)}")
        raise

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
    try:
        # Get duration for initial estimate
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        
        # Test first chunk to determine chunk size
        with tempfile.NamedTemporaryFile(suffix='.mp3') as test_file:
            stream = ffmpeg.input(file_path, ss=0, t=OPTIMAL_DURATION)
            stream = ffmpeg.output(stream, test_file.name, 
                acodec='libmp3lame', 
                b='192k',
                loglevel='error'
            )
            ffmpeg.run(stream, overwrite_output=True)
            use_smaller_chunks = os.path.getsize(test_file.name) >= MAX_SEGMENT_SIZE
        
        # Calculate segments based on test
        chunk_duration = FALLBACK_DURATION if use_smaller_chunks else OPTIMAL_DURATION
        estimated_segments = max(1, int(duration / chunk_duration) + 1)
        
        # Initialize progress
        update_progress(session_id, 0, estimated_segments)
        
        segments_processed = 0
        transcriptions = []
        current_segment = None
        
        # Process segments
        for segment_path in split_audio_streaming(file_path):
            try:
                if current_segment and os.path.exists(current_segment):
                    os.unlink(current_segment)
                
                current_segment = segment_path
                transcription = transcribe_segment(segment_path)
                transcriptions.append(transcription)
                segments_processed += 1
                
                update_progress(session_id, segments_processed, estimated_segments)
                
            except Exception as e:
                logging.error(f"Error processing segment: {str(e)}")
                raise
        
        if current_segment and os.path.exists(current_segment):
            os.unlink(current_segment)
            
        return ' '.join(transcriptions)
            
    except Exception as e:
        logging.error(f"Error in process_audio_file: {str(e)}")
        raise

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
        
        # Save file and get size
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
        # Initialize progress with estimated segments immediately
        estimated_segments = estimate_segments(file_size)
        update_progress(session_id, 0, estimated_segments)
        
        # Start processing in background
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
