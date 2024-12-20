import os
import logging
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from openai import OpenAI
import tempfile
from threading import Lock
import ffmpeg
from audio_processor import AudioProcessor
import prompts
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

# Progress tracking
audio_processor = AudioProcessor()

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
        return transcript.text
        
    except Exception as e:
        logging.error(f"Error in transcribe_segment: {str(e)}")
        raise

def process_audio_file(file_path, session_id):
    """Process audio file with progress tracking"""
    try:
        audio_processor.update_progress(session_id, 0, 1, status="Analyzing audio file...")
        
        # Get duration and segment information
        duration, estimated_segments, _ = audio_processor.get_audio_duration(file_path)
        minutes = int(duration / 60)
        
        audio_processor.update_progress(session_id, 0, estimated_segments, 
                       status=f"Starting transcription... (about {minutes} minute{'s' if minutes != 1 else ''} of audio)")
        
        segments_processed = 0
        transcriptions = []
        current_segment = None
        
        for segment_path in audio_processor.split_audio_streaming(file_path):
            try:
                if current_segment and os.path.exists(current_segment):
                    os.unlink(current_segment)
                
                current_segment = segment_path
                transcription = transcribe_segment(segment_path)
                transcriptions.append(transcription)
                segments_processed += 1
                
                progress_msg = f"Transcribing audio... ({segments_processed} of {estimated_segments} parts complete)"
                if segments_processed == estimated_segments:
                    progress_msg = "Finalizing transcription..."
                
                audio_processor.update_progress(session_id, segments_processed, estimated_segments, status=progress_msg)
                
            except Exception as e:
                logging.error(f"Error processing segment: {str(e)}")
                raise
        
        if current_segment and os.path.exists(current_segment):
            os.unlink(current_segment)
            
        return ' '.join(transcriptions)
            
    except Exception as e:
        logging.error(f"Error in process_audio_file: {str(e)}")
        with audio_processor.progress_lock:
            audio_processor.progress_data[session_id].update({
                'status': 'Error: Failed to process audio file',
                'error': str(e)
            })
        raise

@app.route('/check-progress/<session_id>')
def check_progress(session_id):
    """Endpoint for checking transcription progress"""
    progress = audio_processor.get_progress(session_id)
    if not progress:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(progress)

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
        
        # Save file
        file.save(file_path)
        
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
            
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
                with audio_processor.progress_lock:
                    audio_processor.progress_data[session_id].update({
                        'status': 'complete',
                        'success': True,
                        'id': post_id
                    })
                    
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                with audio_processor.progress_lock:
                    audio_processor.progress_data[session_id].update({
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
        return render_template('transcripteditor.html', 
                             blog_posts=[app.blog_posts[post_id]])
    except Exception as e:
        logging.error(f"Error rendering blog template: {str(e)}")
        return f"Error rendering blog template: {str(e)}", 500

@app.route('/chat/<int:post_id>', methods=['POST'])
def chat_with_ai(post_id):
    try:
        if not hasattr(app, 'blog_posts') or post_id >= len(app.blog_posts):
            return jsonify({'error': 'Blog post not found'}), 404

        data = request.json
        message = data.get('message')
        blog_content = data.get('blogContent')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Create chat completion with OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # or "gpt-3.5-turbo" depending on your needs
            messages=[
                {"role": "system", "content": "You are a helpful transcript editor. Your task is to help users fix any inaccuracies in transcripts while preserving the original meaning and style. Focus on correcting errors in transcription, grammar, and punctuation while maintaining the speaker's voice and intent."},
                {"role": "user", "content": f"Here's the current blog content:\n\n{blog_content}\n\nUser request: {message}"}
            ]
        )

        # Get the AI's response
        ai_response = response.choices[0].message.content

        # Update the blog post content
        app.blog_posts[post_id]['content'] = ai_response

        return jsonify({'response': ai_response})

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/save-and-next/<int:post_id>', methods=['POST'])
def save_and_next(post_id):
    try:
        if not hasattr(app, 'blog_posts') or post_id >= len(app.blog_posts):
            return jsonify({'error': 'Blog post not found'}), 404

        data = request.json
        content = data.get('content')
        
        if not content:
            return jsonify({'error': 'No content provided'}), 400

        # Save the updated content
        app.blog_posts[post_id]['content'] = content

        # Generate topic cards from the transcript
        topics = generate_topic_cards(content)
        
        # Store topic cards in session or app context
        if not hasattr(app, 'topic_cards'):
            app.topic_cards = {}
        app.topic_cards[post_id] = topics

        # Return the URL for the topic cards page
        return jsonify({
            'success': True,
            'nextUrl': url_for('view_topic_cards', post_id=post_id)
        })

    except Exception as e:
        logging.error(f"Error in save_and_next: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/topic-cards/<int:post_id>')
def view_topic_cards(post_id):
    if not hasattr(app, 'topic_cards') or post_id not in app.topic_cards:
        return "Topic cards not found", 404
    
    try:
        return render_template('topiccards.html', 
                             cards=app.topic_cards[post_id])
    except Exception as e:
        logging.error(f"Error rendering topic cards template: {str(e)}")
        return f"Error rendering topic cards template: {str(e)}", 500

def generate_topic_cards(transcript):
    """Generate topic cards from the transcript using AI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompts.topiccards},
                {"role": "user", "content": f"6 grain {transcript}"}
            ]
        )

        # Process the AI response into structured cards
        ai_response = response.choices[0].message.content
        
        # Split the response into cards (assuming AI formats with clear separators)
        # This is a simple implementation - you might need to adjust based on your AI's output format
        raw_cards = ai_response.split('\n\n')
        
        cards = []
        for raw_card in raw_cards:
            if ':' in raw_card:
                title, content = raw_card.split(':', 1)
                cards.append({
                    'title': title.strip(),
                    'content': content.strip()
                })

        return cards

    except Exception as e:
        logging.error(f"Error generating topic cards: {str(e)}")
        raise

if __name__ == '__main__':
    setup_upload_folder()  # Ensure the uploads directory exists
    app.run(debug=True)
