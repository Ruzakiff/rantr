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
        granularity = data.get('granularity', 3)  # Default to 3 if not specified
        
        if not content:
            return jsonify({'error': 'No content provided'}), 400

        # Save the updated content
        app.blog_posts[post_id]['content'] = content

        # Generate topic cards from the transcript with specified granularity
        topics = generate_topic_cards(content, granularity)
        
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
                             cards=app.topic_cards[post_id],
                             post_id=post_id)
    except Exception as e:
        logging.error(f"Error rendering topic cards template: {str(e)}")
        return f"Error rendering topic cards template: {str(e)}", 500

def generate_topic_cards(transcript, granularity=3):
    """Generate topic cards from the transcript using AI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompts.topiccards},
                {"role": "user", "content": f"{granularity} grain {transcript}"}
            ],
            temperature=0
        )

        # Process the AI response into structured cards
        ai_response = response.choices[0].message.content
        
        # Split by "Topic N:" pattern
        import re
        topics = re.split(r'Topic \d+:', ai_response)[1:]  # Skip the first empty split
        
        cards = []
        for i, topic in enumerate(topics, 1):
            if ':' in topic:
                title, content = topic.split(':', 1)
            else:
                # If no colon in the topic, treat first line as title
                lines = topic.strip().split('\n', 1)
                title = lines[0]
                content = lines[1] if len(lines) > 1 else ""
            
            cards.append({
                'title': f"Topic {i}: {title.strip()}",
                'content': content.strip()
            })

        # Ensure we have exactly the requested number of cards
        if len(cards) != granularity:
            logging.warning(f"Generated {len(cards)} cards instead of requested {granularity}")
            # If we got fewer cards than requested, add placeholder cards
            while len(cards) < granularity:
                cards.append({
                    'title': f"Topic {len(cards) + 1}",
                    'content': "Content needs to be generated for this topic."
                })
            # If we got more cards than requested, truncate
            cards = cards[:granularity]

        return cards

    except Exception as e:
        logging.error(f"Error generating topic cards: {str(e)}")
        raise

@app.route('/merge-topics/<int:post_id>', methods=['POST'])
def merge_topics(post_id):
    try:
        data = request.json
        card_index = data.get('cardIndex')
        merged_content = data.get('mergedContent')

        # Get current cards
        cards = app.topic_cards[post_id]
        
        # Create new merged card
        merged_card = {
            'title': f"Topic {card_index + 1}: Merged Topics",
            'content': merged_content
        }

        # Remove the two cards being merged and insert the merged card
        cards.pop(card_index + 1)
        cards[card_index] = merged_card

        # Renumber remaining cards
        for i, card in enumerate(cards, 1):
            card['title'] = f"Topic {i}: {card['title'].split(':', 1)[1].strip()}"

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/split-topic/<int:post_id>', methods=['POST'])
def split_topic(post_id):
    try:
        data = request.json
        card_index = data.get('cardIndex')
        content = data.get('content')
        
        logging.debug(f"Splitting topic {card_index} with content length: {len(content)}")

        # Get current cards
        if not hasattr(app, 'topic_cards') or post_id not in app.topic_cards:
            logging.error(f"Topic cards not found for post_id: {post_id}")
            return jsonify({'error': 'Topic cards not found'}), 404
            
        cards = app.topic_cards[post_id]
        logging.debug(f"Current number of cards: {len(cards)}")

        # Use same format as generate_topic_cards
        system_prompt = """You are an expert at analyzing content and breaking it down into distinct topics. 
        Split this content into EXACTLY two distinct topics.
        
        Format requirements:
        1. Create EXACTLY 2 cards
        2. Each card must start with "Topic N: " where N is the topic number
        3. Each card must be separated by TWO newlines
        4. Each topic must have a clear title and detailed content
        
        Example format:
        Topic 1: [Clear Title]
        [Detailed content for first topic]

        Topic 2: [Clear Title]
        [Detailed content for second topic]"""

        logging.debug("Calling OpenAI API for split")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Split this content into two clear topics:\n\n{content}"}
            ],
            temperature=0
        )

        split_content = response.choices[0].message.content
        logging.debug(f"AI Response received. Length: {len(split_content)}")
        logging.debug(f"AI Response content: {split_content}")

        # Parse the split topics using the same logic as generate_topic_cards
        import re
        topics = re.split(r'Topic \d+:', split_content)[1:]  # Skip the first empty split
        logging.debug(f"Number of topics split: {len(topics)}")

        if len(topics) != 2:
            logging.error(f"AI generated {len(topics)} topics instead of 2")
            return jsonify({'error': 'AI did not generate exactly two topics'}), 500

        # Remove the original card
        cards.pop(card_index)
        
        # Insert the two new cards
        for i, topic in enumerate(topics):
            if ':' in topic:
                title, content = topic.split(':', 1)
            else:
                lines = topic.strip().split('\n', 1)
                title = lines[0]
                content = lines[1] if len(lines) > 1 else ""
            
            new_card = {
                'title': f"Topic {card_index + i + 1}: {title.strip()}",
                'content': content.strip()
            }
            logging.debug(f"Adding new card: {new_card['title']}")
            cards.insert(card_index + i, new_card)

        # Renumber all cards
        for i, card in enumerate(cards, 1):
            card['title'] = f"Topic {i}: {card['title'].split(':', 1)[1].strip() if ':' in card['title'] else card['title']}"
        
        logging.debug(f"Final number of cards: {len(cards)}")
        app.topic_cards[post_id] = cards  # Ensure we save the updated cards

        return jsonify({
            'success': True,
            'message': 'Topic successfully split'
        })

    except Exception as e:
        logging.error(f"Error in split_topic: {str(e)}")
        logging.exception("Full traceback:")  # This will log the full stack trace
        return jsonify({'error': str(e)}), 500

@app.route('/exclude-topic/<int:post_id>', methods=['POST'])
def exclude_topic(post_id):
    try:
        data = request.json
        card_index = data.get('cardIndex')

        # Get current cards
        cards = app.topic_cards[post_id]
        
        # Remove the card
        cards.pop(card_index)

        # Renumber remaining cards
        for i, card in enumerate(cards, 1):
            card['title'] = f"Topic {i}: {card['title'].split(':', 1)[1].strip()}"

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    setup_upload_folder()  # Ensure the uploads directory exists
    app.run(debug=True)
