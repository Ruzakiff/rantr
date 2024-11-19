import os
import logging
import tempfile
import ffmpeg
from threading import Lock

class AudioProcessor:
    def __init__(self, max_segment_size=24*1024*1024, optimal_duration=180, fallback_duration=90):
        self.MAX_SEGMENT_SIZE = max_segment_size
        self.OPTIMAL_DURATION = optimal_duration
        self.FALLBACK_DURATION = fallback_duration
        self.progress_data = {}
        self.progress_lock = Lock()

    def update_progress(self, session_id, current, total, status=None):
        """Update progress for a given session with thread safety"""
        with self.progress_lock:
            self.progress_data[session_id] = {
                'current': current,
                'total': total,
                'percentage': int((current / total) * 100) if total > 0 else 0,
                'status': status or 'Processing...'
            }

    def get_progress(self, session_id):
        """Get progress for a session"""
        with self.progress_lock:
            return self.progress_data.get(session_id)

    def split_audio_streaming(self, file_path):
        """Split audio file into chunks, streaming with ffmpeg"""
        try:
            probe = ffmpeg.probe(file_path)
            duration = float(probe['format']['duration'])
            
            # Test first chunk
            with tempfile.NamedTemporaryFile(suffix='.mp3') as test_file:
                stream = ffmpeg.input(file_path, ss=0, t=self.OPTIMAL_DURATION)
                stream = ffmpeg.output(stream, test_file.name, 
                    acodec='libmp3lame', 
                    b='192k',
                    loglevel='error'
                )
                ffmpeg.run(stream, overwrite_output=True)
                use_smaller_chunks = os.path.getsize(test_file.name) >= self.MAX_SEGMENT_SIZE
            
            chunk_duration = self.FALLBACK_DURATION if use_smaller_chunks else self.OPTIMAL_DURATION
            logging.info(f"Using {chunk_duration}s chunks based on initial test")
            
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
                        
                        if os.path.getsize(temp_file.name) >= self.MAX_SEGMENT_SIZE:
                            os.unlink(temp_file.name)
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

    def get_audio_duration(self, file_path):
        """Get audio file duration and determine chunk size"""
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        
        with tempfile.NamedTemporaryFile(suffix='.mp3') as test_file:
            stream = ffmpeg.input(file_path, ss=0, t=self.OPTIMAL_DURATION)
            stream = ffmpeg.output(stream, test_file.name, 
                acodec='libmp3lame', 
                b='192k',
                loglevel='error'
            )
            ffmpeg.run(stream, overwrite_output=True)
            use_smaller_chunks = os.path.getsize(test_file.name) >= self.MAX_SEGMENT_SIZE
        
        chunk_duration = self.FALLBACK_DURATION if use_smaller_chunks else self.OPTIMAL_DURATION
        estimated_segments = max(1, int(duration / chunk_duration) + 1)
        
        return duration, estimated_segments, chunk_duration