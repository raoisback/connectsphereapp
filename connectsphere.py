import os
import json
from datetime import datetime, timedelta
from time import time
import random
import bcrypt
import sounddevice as sd
import numpy as np
import soundfile as sf
from kivy.app import App
from kivy.uix.screen import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.behaviors import DragBehavior
from kivy.uix.filechooser import FileChooserIconView
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ObjectProperty, ListProperty, DictProperty
from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRoundFlatButton
from kivymd.uix.textfield import MDTextField
from kivy.storage.jsonstore import JsonStore
from kivy.uix.image import AsyncImage
from kivy.logger import Logger
from kivy.lang import Builder
from kivy.metrics import dp
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image as PILImage
import io
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import re
import cv2
import mediapipe as mp
from kivy.core.image import Image as CoreImage
from kivy.core.audio import SoundLoader
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.behaviors.button import ButtonBehavior

# Initialize Firebase
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cred = credentials.Certificate(os.path.join(BASE_DIR, 'serviceAccountKey.json'))
firebase_admin.initialize_app(cred, {'storageBucket': 'connectsphere-d979f.appspot.com'})
db = firestore.client()
bucket = storage.bucket()

# Asset paths
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)
    Logger.warning("Assets directory created as it was missing.")

class ImageHelper:
    @staticmethod
    def compress_image(image_path, max_size=(800, 800), quality=85):
        try:
            img = PILImage.open(image_path)
            img.thumbnail(max_size, PILImage.LANCZOS)
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            return output.getvalue()
        except Exception as e:
            Logger.error(f"Image compression error: {e}")
            return None

class CacheManager:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._image_cache = {}
        self._max_cache_size = 100 * 1024 * 1024  # 100MB cache limit
        
    @lru_cache(maxsize=100)
    def get_profile(self, user_id):
        return backend.get_profile(user_id)
        
    @lru_cache(maxsize=1000)
    def get_post(self, post_id):
        return backend.db.collection('posts').document(post_id.split('_')[-1]).get()

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self.get_profile.cache_clear()
            self.get_post.cache_clear()

    def cache_image(self, url, image_data):
        with self._lock:
            if sum(len(data) for data in self._image_cache.values()) > self._max_cache_size:
                # Remove oldest items
                while self._image_cache and sum(len(data) for data in self._image_cache.values()) > self._max_cache_size * 0.8:
                    self._image_cache.pop(next(iter(self._image_cache)))
            self._image_cache[url] = image_data

    def get_cached_image(self, url):
        return self._image_cache.get(url)

cache_manager = CacheManager()

class RetryableOperation:
    def __init__(self, operation, max_retries=3, delay=1):
        self.operation = operation
        self.max_retries = max_retries
        self.delay = delay

    def execute(self, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return self.operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.delay)
                continue

class AIHelper:
    def __init__(self):
        self.content_classifier = None
        self.sentiment_analyzer = None
        self.caption_generator = None
        
    def moderate_content(self, text, image_url=None):
        try:
            # Basic content moderation
            inappropriate_words = ['hate', 'violence', 'abuse', 'spam']
            return not any(word in text.lower() for word in inappropriate_words)
        except Exception as e:
            Logger.error(f"AIHelper moderate_content error: {e}")
            return True
            
    def analyze_mood(self, text):
        try:
            # Simple sentiment analysis
            positive_words = ['happy', 'great', 'love', 'joy', 'excited']
            negative_words = ['sad', 'angry', 'hate', 'upset']
            
            pos_score = sum(word in text.lower() for word in positive_words)
            neg_score = sum(word in text.lower() for word in negative_words)
            
            if pos_score > neg_score:
                return 'positive'
            elif neg_score > pos_score:
                return 'negative'
            return 'neutral'
        except Exception as e:
            Logger.error(f"AIHelper analyze_mood error: {e}")
            return 'neutral'
            
    def suggest_caption(self, image_url):
        try:
            # Basic caption suggestions based on time and activity
            time_of_day = datetime.now().hour
            if 5 <= time_of_day < 12:
                return "Starting my day with positive vibes! ☀️"
            elif 12 <= time_of_day < 17:
                return "Making the most of this beautiful afternoon! 🌤️"
            else:
                return "Evening reflections and peaceful moments ✨"
        except Exception as e:
            Logger.error(f"AIHelper suggest_caption error: {e}")
            return "Share your thoughts..."
            
    def recommend_content(self, user_id, content_type='post'):
        try:
            # Get user's recent interactions
            user_likes = backend.get_user_likes(user_id)
            user_views = backend.get_user_views(user_id)
            
            # Basic content recommendation
            recommended = []
            all_posts = backend.get_all_posts()
            
            for post in all_posts:
                score = 0
                # Higher score for posts similar to liked content
                if post['content'] in user_likes:
                    score += 2
                # Boost score for posts with similar topics
                if any(topic in post['content'].lower() for topic in user_views):
                    score += 1
                if score > 0:
                    recommended.append((post, score))
                    
            return sorted(recommended, key=lambda x: x[1], reverse=True)[:5]
        except Exception as e:
            Logger.error(f"AIHelper recommend_content error: {e}")
            return []

ai_helper = AIHelper()

class DatabaseBackend:
    def __init__(self):
        self.db = db
        self.bucket = bucket
        self.collaborative_sessions = {}
        self.ar_markers = {}

    def register_user(self, username, password, email="", privacy="public"):
        try:
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user_ref = self.db.collection('users').document(username)
            if user_ref.get().exists:
                return False
            user_ref.set({
                'user_id': username,
                'username': username,
                'password': hashed_pw.decode('utf-8'),
                'email': email,
                'bio': '',
                'profile_pic': '',
                'privacy': privacy,
                'analytics': {'post_count': 0, 'like_count': 0, 'view_count': 0}
            })
            return True
        except Exception as e:
            Logger.error(f"register_user error: {e}")
            return False

    def login_user(self, username, password):
        try:
            user_ref = self.db.collection('users').document(username)
            user = user_ref.get()
            if user.exists and bcrypt.checkpw(password.encode('utf-8'), user.to_dict()['password'].encode('utf-8')):
                return True
            return False
        except Exception as e:
            Logger.error(f"login_user error: {e}")
            return False

    def update_profile(self, user_id, bio=None, profile_pic=None, privacy=None):
        try:
            user_ref = self.db.collection('users').document(user_id)
            updates = {}
            if bio is not None:
                updates['bio'] = bio
            if profile_pic is not None:
                blob = self.bucket.blob(f'profile_pics/{user_id}/{os.path.basename(profile_pic)}')
                blob.upload_from_filename(profile_pic)
                blob.make_public()
                updates['profile_pic'] = blob.public_url
            if privacy is not None:
                updates['privacy'] = privacy
            user_ref.update(updates)
            return True
        except Exception as e:
            Logger.error(f"update_profile error: {e}")
            return False

    def get_profile(self, user_id):
        try:
            user_ref = self.db.collection('users').document(user_id)
            user = user_ref.get()
            if user.exists:
                data = user.to_dict()
                return {"username": data['username'], "bio": data.get('bio', ''), "profile_pic": data.get('profile_pic', ''), "privacy": data.get('privacy', 'public'), "analytics": data.get('analytics', {})}
            return None
        except Exception as e:
            Logger.error(f"get_profile error: {e}")
            return None

    def share_post(self, feature, content, user_id, image_url=None, collaborators=None):
        operation = RetryableOperation(self._share_post)
        return operation.execute(feature, content, user_id, image_url, collaborators)

    def _share_post(self, feature, content, user_id, image_url=None, collaborators=None):
        try:
            timestamp = time()
            if feature == 'orbit':
                orbit, content = content.split(':', 1)
                feature_key = f'orbit_{orbit}'
            else:
                feature_key = feature
            post_ref = self.db.collection('posts').document()
            post_data = {
                'user_id': user_id,
                'feature': feature_key,
                'content': content,
                'timestamp': timestamp,
                'likes': 0,
                'views': 0,
                'image_url': image_url if image_url else '',
                'collaborators': collaborators if collaborators else [],
                'reported': False
            }
            if image_url:
                compressed_image = ImageHelper.compress_image(image_url)
                if compressed_image:
                    blob = bucket.blob(f'posts/{user_id}/{int(time())}.jpg')
                    blob.upload_from_string(compressed_image, content_type='image/jpeg')
                    blob.make_public()
                    image_url = blob.public_url
            post_ref.set(post_data)
            if feature == 'glow' and any(word in content.lower() for word in ['happy', 'great', 'love', 'awesome', 'good', 'wonderful', 'amazing']):
                points_ref = self.db.collection('points').document(user_id)
                points = points_ref.get()
                new_points = (points.to_dict().get('points', 0) if points.exists else 0) + 10
                points_ref.set({'user_id': user_id, 'points': new_points})
            user_ref = self.db.collection('users').document(user_id)
            user_ref.update({'analytics.post_count': firestore.Increment(1)})
            return f"{feature_key}_{post_ref.id}"
        except Exception as e:
            Logger.error(f"share_post error: {e}")
            return None

    def get_posts(self, feature, orbit=None, limit=10, offset=0):
        try:
            feature_key = f'orbit_{orbit}' if feature == 'orbit' and orbit else feature
            query = self.db.collection('posts').where('feature', '==', feature_key)
            if feature == 'pulse':
                query = query.where('timestamp', '>', time() - 24 * 3600)
            posts = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).offset(offset).get()
            return [{'id': f"{feature_key}_{post.id}", 'user_id': post.to_dict()['user_id'], 'content': post.to_dict()['content'], 'timestamp': post.to_dict()['timestamp'], 'image_url': post.to_dict().get('image_url', ''), 'collaborators': post.to_dict().get('collaborators', [])} for post in posts]
        except Exception as e:
            Logger.error(f"get_posts error: {e}")
            return []

    def listen_to_posts(self, feature, callback, orbit=None):
        feature_key = f'orbit_{orbit}' if feature == 'orbit' and orbit else feature
        query = self.db.collection('posts').where('feature', '==', feature_key)
        return query.on_snapshot(lambda docs, changes, read_time: callback(docs))

    def add_comment(self, post_id, user_id, content):
        try:
            timestamp = time()
            post_id_num = post_id.split('_')[-1]
            comment_ref = self.db.collection('comments').document()
            comment_ref.set({
                'post_id': post_id_num,
                'user_id': user_id,
                'content': content,
                'timestamp': timestamp
            })
            post_ref = self.db.collection('posts').document(post_id_num)
            post = post_ref.get()
            if post.exists:
                notif_ref = self.db.collection('notifications').document()
                notif_ref.set({
                    'user_id': post.to_dict()['user_id'],
                    'content': f"{user_id} commented on your post",
                    'timestamp': timestamp,
                    'read': False
                })
            return True
        except Exception as e:
            Logger.error(f"add_comment error: {e}")
            return False

    def get_comments(self, post_id):
        try:
            post_id_num = post_id.split('_')[-1]
            comments = self.db.collection('comments').where('post_id', '==', post_id_num).order_by('timestamp').get()
            return [{'user_id': comment.to_dict()['user_id'], 'content': comment.to_dict()['content'], 'timestamp': comment.to_dict()['timestamp']} for comment in comments]
        except Exception as e:
            Logger.error(f"get_comments error: {e}")
            return []

    def send_message(self, sender_id, receiver_id, content):
        try:
            timestamp = time()
            message_ref = self.db.collection('messages').document()
            message_ref.set({
                'sender_id': sender_id,
                'receiver_id': receiver_id,
                'content': content,
                'timestamp': timestamp
            })
            notif_ref = self.db.collection('notifications').document()
            notif_ref.set({
                'user_id': receiver_id,
                'content': f"{sender_id} sent you a message",
                'timestamp': timestamp,
                'read': False
            })
            return True
        except Exception as e:
            Logger.error(f"send_message error: {e}")
            return False

    def get_messages(self, user_id, other_user_id):
        try:
            messages = self.db.collection('messages').where('sender_id', 'in', [user_id, other_user_id]).where('receiver_id', 'in', [user_id, other_user_id]).order_by('timestamp').get()
            return [{'sender_id': msg.to_dict()['sender_id'], 'receiver_id': msg.to_dict()['receiver_id'], 'content': msg.to_dict()['content'], 'timestamp': msg.to_dict()['timestamp']} for msg in messages]
        except Exception as e:
            Logger.error(f"get_messages error: {e}")
            return []

    def listen_to_messages(self, user_id, callback):
        query = self.db.collection('messages').where('receiver_id', '==', user_id)
        return query.on_snapshot(lambda docs, changes, read_time: callback(docs))

    def get_notifications(self, user_id):
        try:
            notifs = self.db.collection('notifications').where('user_id', '==', user_id).order_by('timestamp', direction=firestore.Query.DESCENDING).get()
            return [{'content': notif.to_dict()['content'], 'timestamp': notif.to_dict()['timestamp'], 'read': notif.to_dict()['read']} for notif in notifs]
        except Exception as e:
            Logger.error(f"get_notifications error: {e}")
            return []

    def listen_to_notifications(self, user_id, callback):
        query = self.db.collection('notifications').where('user_id', '==', user_id)
        return query.on_snapshot(lambda docs, changes, read_time: callback(docs))

    def mark_notification_read(self, user_id, timestamp):
        try:
            notifs = self.db.collection('notifications').where('user_id', '==', user_id).where('timestamp', '==', timestamp).get()
            for notif in notifs:
                notif.reference.update({'read': True})
            return True
        except Exception as e:
            Logger.error(f"mark_notification_read error: {e}")
            return False

    def toggle_like(self, post_id, user_id):
        try:
            post_id_num = post_id.split('_')[-1]
            post_ref = self.db.collection('posts').document(post_id_num)
            post = post_ref.get()
            if post.exists:
                action = random.random() > 0.3
                likes = post.to_dict()['likes'] + (1 if action else -1)
                post_ref.update({'likes': likes})
                if action:
                    notif_ref = self.db.collection('notifications').document()
                    notif_ref.set({
                        'user_id': post.to_dict()['user_id'],
                        'content': f"{user_id} liked your post",
                        'timestamp': time(),
                        'read': False
                    })
                    self.db.collection('users').document(post.to_dict()['user_id']).update({'analytics.like_count': firestore.Increment(1)})
                return True
            return False
        except Exception as e:
            Logger.error(f"toggle_like error: {e}")
            return False

    def get_like_count(self, post_id):
        try:
            post_id_num = post_id.split('_')[-1]
            post = self.db.collection('posts').document(post_id_num).get()
            return post.to_dict()['likes'] if post.exists else 0
        except Exception as e:
            Logger.error(f"get_like_count error: {e}")
            return 0

    def toggle_follow(self, user_id, target_id):
        try:
            follow_ref = self.db.collection('follows').document(f"{user_id}_{target_id}")
            if follow_ref.get().exists:
                follow_ref.delete()
            else:
                follow_ref.set({'follower_id': user_id, 'following_id': target_id})
                notif_ref = self.db.collection('notifications').document()
                notif_ref.set({
                    'user_id': target_id,
                    'content': f"{user_id} started following you",
                    'timestamp': time(),
                    'read': False
                })
            return True
        except Exception as e:
            Logger.error(f"toggle_follow error: {e}")
            return False

    def get_followers(self, user_id):
        try:
            followers = self.db.collection('follows').where('following_id', '==', user_id).get()
            return len(followers)
        except Exception as e:
            Logger.error(f"get_followers error: {e}")
            return 0

    def get_following(self, user_id):
        try:
            following = self.db.collection('follows').where('follower_id', '==', user_id).get()
            return len(following)
        except Exception as e:
            Logger.error(f"get_following error: {e}")
            return 0

    def get_points(self, user_id):
        try:
            points = self.db.collection('points').document(user_id).get()
            return points.to_dict()['points'] if points.exists else 0
        except Exception as e:
            Logger.error(f"get_points error: {e}")
            return 0

    def schedule_capsule(self, user_id, content, unlock_time):
        try:
            capsule_ref = self.db.collection('time_capsules').document()
            capsule_ref.set({
                'user_id': user_id,
                'content': content,
                'unlock_time': unlock_time,
                'unlocked': False
            })
            threading.Timer(unlock_time - time(), lambda: self.unlock_capsule(capsule_ref.id)).start()
            return capsule_ref.id
        except Exception as e:
            Logger.error(f"schedule_capsule error: {e}")
            return None

    def unlock_capsule(self, capsule_id):
        try:
            capsule_ref = self.db.collection('time_capsules').document(capsule_id)
            capsule = capsule_ref.get()
            if capsule.exists and not capsule.to_dict()['unlocked']:
                capsule_ref.update({'unlocked': True})
                backend.share_post('capsule', capsule.to_dict()['content'], capsule.to_dict()['user_id'])
                notif_ref = self.db.collection('notifications').document()
                notif_ref.set({
                    'user_id': capsule.to_dict()['user_id'],
                    'content': f"Your time capsule '{capsule.to_dict()['content'][:20]}...' is now unlocked!",
                    'timestamp': time(),
                    'read': False
                })
        except Exception as e:
            Logger.error(f"unlock_capsule error: {e}")

    def get_capsules(self, user_id):
        try:
            capsules = self.db.collection('time_capsules').where('user_id', '==', user_id).get()
            return [{'id': capsule.id, 'content': capsule.to_dict()['content'], 'unlock_time': capsule.to_dict()['unlock_time'], 'unlocked': capsule.to_dict()['unlocked']} for capsule in capsules]
        except Exception as e:
            Logger.error(f"get_capsules error: {e}")
            return []

    def search_users(self, query):
        try:
            users = self.db.collection('users').get()
            return [user.to_dict()['username'] for user in users if query.lower() in user.to_dict()['username'].lower() and user.to_dict()['privacy'] == 'public']
        except Exception as e:
            Logger.error(f"search_users error: {e}")
            return []

    def search_posts(self, query):
        try:
            words = query.lower().split()
            posts = self.db.collection('posts').get()
            results = []
            for post in posts:
                post_data = post.to_dict()
                content = post_data['content'].lower()
                if any(word in content for word in words):
                    results.append({
                        'id': post.id,
                        'user_id': post_data['user_id'],
                        'content': post_data['content'],
                        'timestamp': post_data['timestamp'],
                        'relevance': sum(content.count(word) for word in words)
                    })
            return sorted(results, key=lambda x: x['relevance'], reverse=True)
        except Exception as e:
            Logger.error(f"search_posts error: {e}")
            return []

    def report_post(self, post_id, user_id, reason):
        try:
            post_id_num = post_id.split('_')[-1]
            post_ref = self.db.collection('posts').document(post_id_num)
            post_ref.update({'reported': True})
            report_ref = self.db.collection('reports').document()
            report_ref.set({
                'post_id': post_id_num,
                'user_id': user_id,
                'reason': reason,
                'timestamp': time()
            })
            return True
        except Exception as e:
            Logger.error(f"report_post error: {e}")
            return False

    def increment_views(self, post_id):
        try:
            post_id_num = post_id.split('_')[-1]
            post_ref = self.db.collection('posts').document(post_id_num)
            post_ref.update({'views': firestore.Increment(1)})
            user_id = post_ref.get().to_dict()['user_id']
            self.db.collection('users').document(user_id).update({'analytics.view_count': firestore.Increment(1)})
        except Exception as e:
            Logger.error(f"increment_views error: {e}")

    def create_3d_post(self, user_id, model_data, position, orientation):
        try:
            post_ref = self.db.collection('3d_posts').document()
            post_ref.set({
                'user_id': user_id,
                'model_data': model_data,
                'position': position,
                'orientation': orientation,
                'timestamp': time(),
                'views': 0,
                'interactions': 0
            })
            return post_ref.id
        except Exception as e:
            Logger.error(f"DatabaseBackend create_3d_post error: {e}")
            return None

    def start_collaborative_session(self, post_id, user_id):
        try:
            if post_id not in self.collaborative_sessions:
                self.collaborative_sessions[post_id] = set()
            self.collaborative_sessions[post_id].add(user_id)
            return True
        except Exception as e:
            Logger.error(f"DatabaseBackend start_collaborative_session error: {e}")
            return False

    def register_ar_marker(self, marker_id, content_id):
        try:
            self.ar_markers[marker_id] = content_id
            self.db.collection('ar_markers').document(str(marker_id)).set({
                'content_id': content_id,
                'timestamp': time()
            })
            return True
        except Exception as e:
            Logger.error(f"DatabaseBackend register_ar_marker error: {e}")
            return False

    def get_user_likes(self, user_id):
        try:
            likes = self.db.collection('likes').where('user_id', '==', user_id).get()
            return [like.to_dict()['post_content'] for like in likes]
        except Exception as e:
            Logger.error(f"get_user_likes error: {e}")
            return []
            
    def get_user_views(self, user_id):
        try:
            views = self.db.collection('views').where('user_id', '==', user_id).get()
            return [view.to_dict()['post_content'] for view in views]
        except Exception as e:
            Logger.error(f"get_user_views error: {e}")
            return []

backend = DatabaseBackend()

class Story:
    def __init__(self, user_id, media_url, duration=5):
        self.user_id = user_id
        self.media_url = media_url
        self.duration = duration
        self.timestamp = datetime.now()
        self.expires = self.timestamp + timedelta(hours=24)
        self.views = []

class ARManager:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh()
        self.camera = None
        self.frame = None
        self.markers = {}
        self.frame_processor = None
        self.ar_objects = {}
        self.last_hand_gesture = None
        self.last_face_position = None
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
    def start(self):
        try:
            self.camera = cv2.VideoCapture(0)
            self.frame_processor = Clock.schedule_interval(self.process_frame, 1.0/30.0)
            Logger.info("ARManager: Started camera and frame processing")
        except Exception as e:
            Logger.error(f"ARManager start error: {e}")

    def stop(self):
        try:
            if self.frame_processor:
                self.frame_processor.cancel()
            if self.camera:
                self.camera.release()
            self.camera = None
            self.last_hand_gesture = None
            self.last_face_position = None
            Logger.info("ARManager: Stopped camera and frame processing")
        except Exception as e:
            Logger.error(f"ARManager stop error: {e}")

    def process_frame(self, dt):
        try:
            if not self.camera or not self.camera.isOpened():
                return None

            ret, frame = self.camera.read()
            if not ret:
                return None

            # Convert the frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Process hands
            hand_results = self.hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                frame.flags.writeable = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.drawing_styles.get_default_hand_landmarks_style(),
                        self.drawing_styles.get_default_hand_connections_style()
                    )
                    # Process gestures
                    gesture = self.recognize_gesture(hand_landmarks)
                    if gesture != self.last_hand_gesture:
                        self.last_hand_gesture = gesture
                        self.handle_gesture(gesture)

            # Process face
            face_results = self.face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                frame.flags.writeable = True
                for face_landmarks in face_results.multi_face_landmarks:
                    # Draw face mesh
                    self.drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_styles.get_default_face_mesh_contours_style()
                    )
                    # Track face position
                    face_position = self.get_face_position(face_landmarks)
                    if face_position != self.last_face_position:
                        self.last_face_position = face_position
                        self.update_ar_effects(face_position)

            # Render AR objects
            frame = self.render_ar_objects(frame)

            # Convert frame to texture for Kivy
            texture = self.frame_to_texture(frame)
            return texture

        except Exception as e:
            Logger.error(f"ARManager process_frame error: {e}")
            return None

    def recognize_gesture(self, landmarks):
        try:
            # Extract key points
            thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # Calculate distances
            thumb_index_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            index_middle_dist = ((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)**0.5
            
            # Recognize gestures
            if thumb_index_dist < 0.1:
                return "pinch"
            elif index_middle_dist < 0.1:
                return "peace"
            elif thumb_tip.y < index_tip.y:
                return "thumbs_up"
            return None
        except Exception as e:
            Logger.error(f"ARManager recognize_gesture error: {e}")
            return None

    def get_face_position(self, landmarks):
        try:
            # Get nose position as reference
            nose = landmarks.landmark[1]
            return {'x': nose.x, 'y': nose.y, 'z': nose.z}
        except Exception as e:
            Logger.error(f"ARManager get_face_position error: {e}")
            return None

    def handle_gesture(self, gesture):
        try:
            if gesture == "pinch":
                # Handle pinch gesture (e.g., grab AR object)
                Logger.info("ARManager: Pinch gesture detected")
            elif gesture == "peace":
                # Handle peace gesture (e.g., create AR object)
                Logger.info("ARManager: Peace gesture detected")
            elif gesture == "thumbs_up":
                # Handle thumbs up gesture (e.g., like current content)
                Logger.info("ARManager: Thumbs up gesture detected")
        except Exception as e:
            Logger.error(f"ARManager handle_gesture error: {e}")

    def update_ar_effects(self, face_position):
        try:
            # Update AR effects based on face position
            for obj_id in self.ar_objects:
                # Adjust object position relative to face
                self.ar_objects[obj_id]['position'] = {
                    'x': face_position['x'],
                    'y': face_position['y'] + 0.2,  # Place above head
                    'z': face_position['z']
                }
        except Exception as e:
            Logger.error(f"ARManager update_ar_effects error: {e}")

    def render_ar_objects(self, frame):
        try:
            height, width = frame.shape[:2]
            for obj_id, obj_data in self.ar_objects.items():
                # Convert 3D position to 2D screen coordinates
                x = int(obj_data['position']['x'] * width)
                y = int(obj_data['position']['y'] * height)
                
                # Draw AR object (simple rectangle for now)
                cv2.rectangle(frame, (x-20, y-20), (x+20, y+20), (0, 255, 0), 2)
                
                # Add text label
                cv2.putText(frame, obj_id, (x-10, y-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return frame
        except Exception as e:
            Logger.error(f"ARManager render_ar_objects error: {e}")
            return frame

    def frame_to_texture(self, frame):
        try:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            return texture
        except Exception as e:
            Logger.error(f"ARManager frame_to_texture error: {e}")
            return None

    def add_ar_object(self, object_id, object_data):
        self.ar_objects[object_id] = object_data

    def remove_ar_object(self, object_id):
        if object_id in self.ar_objects:
            del self.ar_objects[object_id]

class DraggableImage(DragBehavior, AsyncImage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drag_rectangle = None
        self.drag_timeout = 1000000
        self.drag_distance = 20
        with self.canvas.before:
            Color(1, 1, 1, 0.5)
            self.drag_rectangle = Rectangle(pos=self.pos, size=self.size)
        if not os.path.exists(self.source):
            self.source = os.path.join(ASSETS_DIR, 'default_icon.png')
            Logger.warning(f"DraggableImage source missing, using default: {self.source}")

    def on_pos(self, *args):
        if self.drag_rectangle:
            self.drag_rectangle.pos = self.pos

    def on_size(self, *args):
        if self.drag_rectangle:
            self.drag_rectangle.size = self.size

class BaseScreen(Screen):
    offset = NumericProperty(0)
    limit = NumericProperty(10)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transition = None
        self.particles = []
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.is_loading = False
        default_bg = os.path.join(ASSETS_DIR, 'default_background.jpg')
        if not os.path.exists(default_bg):
            Logger.warning(f"Background image missing: {default_bg}")
            default_bg = None
        with self.canvas.before:
            self.bg_color = Color(0.1, 0.1, 0.1, 1)
            self.bg_image = AsyncImage(source=default_bg, allow_stretch=True, keep_ratio=False)
            self.bg_image.size = self.size
            self.bg_image.pos = self.pos
            self.gradient = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_layout, size=self.update_layout)
        self.listeners = []
        # Add AR capability
        self.ar_mode = False
        self.ar_session = None
        self.ar_manager = ARManager()

    def update_layout(self, *args):
        try:
            self.bg_image.size = self.size
            self.bg_image.pos = self.pos
            self.gradient.pos = self.pos
            self.gradient.size = self.size
            for p in self.particles:
                p.pos = (random.randint(0, self.width), random.randint(0, self.height))
        except AttributeError as e:
            Logger.error(f"BaseScreen update_layout error: {e}")

    def on_enter(self):
        try:
            if self.transition:
                self.transition.stop()
            self.transition = Animation(pos=(-self.width * 0.2, 0), duration=0.4, t='out_quad') + Animation(pos=(0, 0), duration=0.4, t='out_quad')
            self.transition.start(self)
            self.start_particles()
        except Exception as e:
            Logger.error(f"BaseScreen on_enter error: {e}")

    def on_pre_leave(self):
        try:
            if self.transition:
                self.transition.stop()
            self.transition = Animation(pos=(self.width * 0.2, 0), duration=0.4, t='out_quad')
            self.transition.start(self)
            for p in self.particles:
                self.remove_widget(p)
            self.particles.clear()
            for listener in self.listeners:
                listener.unsubscribe()
            self.listeners.clear()
        except Exception as e:
            Logger.error(f"BaseScreen on_pre_leave error: {e}")

    def toggle_nav_drawer(self):
        try:
            app = App.get_running_app()
            if hasattr(app.root.ids, 'nav_drawer'):
                app.root.ids.nav_drawer.set_state("toggle")
            else:
                Logger.error("Navigation drawer not found in root.ids")
        except AttributeError as e:
            Logger.error(f"BaseScreen toggle_nav_drawer error: {e}")

    def show_search(self):
        try:
            popup = Popup(title='Search', size_hint=(0.8, 0.8))
            layout = BoxLayout(orientation='vertical', padding=10)
            search_input = MDTextField(hint_text="Search users or posts...", mode="rectangle")
            result_box = BoxLayout(orientation='vertical', size_hint_y=0.8)
            layout.add_widget(search_input)
            layout.add_widget(result_box)
            layout.add_widget(MDRoundFlatButton(text="Search", pos_hint={'center_x': 0.5}, on_press=lambda x: self.perform_search(search_input.text, result_box)))
            popup.content = layout
            popup.open()
        except Exception as e:
            Logger.error(f"BaseScreen show_search error: {e}")

    def perform_search(self, query, result_box):
        try:
            result_box.clear_widgets()
            users = backend.search_users(query)
            posts = backend.search_posts(query)
            for user in users:
                result_box.add_widget(MDLabel(text=f"User: {user}", theme_text_color="Custom", text_color=(1, 1, 1, 1)))
            for post in posts:
                result_box.add_widget(MDLabel(text=f"Post by {post['user_id']}: {post['content'][:50]}...", theme_text_color="Custom", text_color=(1, 1, 1, 1)))
            if not users and not posts:
                result_box.add_widget(MDLabel(text="No results found", theme_text_color="Custom", text_color=(0.7, 0.7, 0.7, 1)))
        except Exception as e:
            Logger.error(f"BaseScreen perform_search error: {e}")

    def show_profile(self):
        try:
            app = App.get_running_app()
            popup = Popup(title='Profile', size_hint=(0.8, 0.8))
            layout = BoxLayout(orientation='vertical', padding=10)
            profile = backend.get_profile(app.user_id)
            avatar_path = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
            if not os.path.exists(avatar_path) and not avatar_path.startswith('http'):
                Logger.warning(f"Avatar image missing: {avatar_path}")
                avatar_path = os.path.join(ASSETS_DIR, 'default_icon.png')
            avatar = AsyncImage(source=avatar_path, size_hint=(None, None), size=(100, 100))
            with avatar.canvas.before:
                Color(1, 1, 1, 1)
                Ellipse(pos=avatar.pos, size=avatar.size)
            layout.add_widget(avatar)
            layout.add_widget(MDLabel(text=f"Username: {app.user_id}", halign='center'))
            layout.add_widget(MDLabel(text=f"Bio: {profile['bio'] if profile else ''}", halign='center'))
            posts = db.collection('posts').where('user_id', '==', app.user_id).get()
            post_count = len(posts)
            like_count = sum(post.to_dict()['likes'] for post in posts) if posts else 0
            view_count = profile['analytics']['view_count'] if profile else 0
            layout.add_widget(MDLabel(text=f"Posts: {post_count}", halign='center'))
            layout.add_widget(MDLabel(text=f"Likes: {like_count}", halign='center'))
            layout.add_widget(MDLabel(text=f"Views: {view_count}", halign='center'))
            layout.add_widget(MDLabel(text=f"Followers: {backend.get_followers(app.user_id)}", halign='center'))
            layout.add_widget(MDLabel(text=f"Following: {backend.get_following(app.user_id)}", halign='center'))
            layout.add_widget(MDLabel(text=f"Points: {backend.get_points(app.user_id)}", halign='center'))
            layout.add_widget(MDLabel(text=f"Privacy: {profile['privacy'] if profile else 'public'}", halign='center'))
            layout.add_widget(MDRoundFlatButton(text='Edit Profile', pos_hint={'center_x': 0.5}, on_press=lambda x: self.edit_profile(popup)))
            layout.add_widget(MDRoundFlatButton(text='Logout' if app.logged_in else 'Login', pos_hint={'center_x': 0.5}, on_press=lambda x: self.handle_auth(popup)))
            popup.content = layout
            popup.open()
        except Exception as e:
            Logger.error(f"BaseScreen show_profile error: {e}")

    def edit_profile(self, parent_popup):
        try:
            popup = Popup(title='Edit Profile', size_hint=(0.8, 0.8))
            layout = BoxLayout(orientation='vertical', padding=10)
            bio_input = MDTextField(hint_text="Enter bio", mode="rectangle", multiline=True)
            privacy_spinner = Builder.load_string('Spinner:\n    text: "public"\n    values: ["public", "private"]\n    size_hint_x: 0.3\n    background_color: 0.3, 0.3, 0.3, 0.9\n    color: 1, 1, 1, 1\n    canvas.before:\n        Color:\n            rgba: 0.3, 0.3, 0.3, 0.5\n        RoundedRectangle:\n            pos: self.pos\n            size: self.size\n            radius: [dp(10)]')
            file_chooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg'])
            layout.add_widget(bio_input)
            layout.add_widget(privacy_spinner)
            layout.add_widget(file_chooser)
            submit_btn = MDRoundFlatButton(text='Save', pos_hint={'center_x': 0.5}, on_press=lambda x: self.save_profile(bio_input.text, file_chooser.selection, privacy_spinner.text, popup, parent_popup))
            layout.add_widget(submit_btn)
            popup.content = layout
            popup.open()
        except Exception as e:
            Logger.error(f"BaseScreen edit_profile error: {e}")

    def save_profile(self, bio, files, privacy, popup, parent_popup):
        try:
            app = App.get_running_app()
            profile_pic = files[0] if files else None
            if backend.update_profile(app.user_id, bio=bio, profile_pic=profile_pic, privacy=privacy):
                popup.dismiss()
                parent_popup.dismiss()
                self.show_profile()
        except Exception as e:
            Logger.error(f"BaseScreen save_profile error: {e}")

    def handle_auth(self, popup):
        try:
            app = App.get_running_app()
            if app.logged_in:
                app.logged_in = False
                app.store.put('user', logged_in=False, user_id=app.user_id)
                app.user_id = "Guest"
            else:
                LoginScreen().open()
            popup.dismiss()
        except Exception as e:
            Logger.error(f"BaseScreen handle_auth error: {e}")

    def start_particles(self):
        try:
            for _ in range(10):
                p = Widget()
                with p.canvas:
                    Color(1, 1, 1, 0.2)
                    Ellipse(pos=(random.randint(0, self.width), random.randint(0, self.height)), size=(5, 5))
                anim = Animation(size=(10, 10), opacity=0, duration=2, t='out_quad')
                anim.bind(on_complete=lambda x, w=p: self.remove_particle(w))
                anim.start(p)
                self.particles.append(p)
                self.add_widget(p)
        except Exception as e:
            Logger.error(f"BaseScreen start_particles error: {e}")

    def remove_particle(self, widget):
        try:
            self.remove_widget(widget)
            self.particles.remove(widget)
        except Exception as e:
            Logger.error(f"BaseScreen remove_particle error: {e}")

    def load_more(self, *args):
        if not self.is_loading:
            self.is_loading = True
            self.show_loading()
            self.executor.submit(self._async_load_more)

    def _async_load_more(self):
        try:
            self.offset += self.limit
            Clock.schedule_once(lambda dt: self.update_feed())
        finally:
            self.is_loading = False
            self.hide_loading()

    def show_loading(self):
        spinner = MDSpinner(size_hint=(None, None), size=(48, 48), pos_hint={'center_x': .5, 'center_y': .5})
        self.spinner = spinner
        self.add_widget(spinner)

    def hide_loading(self):
        if hasattr(self, 'spinner'):
            self.remove_widget(self.spinner)

    def toggle_ar(self):
        try:
            if self.ar_mode:
                self.ar_manager.stop()
                self.ids.ar_overlay.opacity = 0
                self.ids.ar_overlay.disabled = True
            else:
                self.ar_manager.start()
                self.ids.ar_overlay.opacity = 1
                self.ids.ar_overlay.disabled = False
            self.ar_mode = not self.ar_mode
        except Exception as e:
            Logger.error(f"BaseScreen toggle_ar error: {e}")

    def update_ar(self, dt):
        try:
            texture = self.ar_manager.get_texture()
            if texture:
                self.ids.ar_overlay.texture = texture
        except Exception as e:
            Logger.error(f"update_ar error: {e}")

class PostCard(MDCard):
    profile_image = StringProperty(os.path.join(ASSETS_DIR, 'default_icon.png'))
    title = StringProperty()
    subtitle = StringProperty()
    timestamp = StringProperty()
    status = StringProperty()
    liked = BooleanProperty(False)
    like_count = NumericProperty(0)
    post_id = StringProperty()
    image_url = StringProperty('')
    reactions = DictProperty({'like': 0, 'heart': 0, 'laugh': 0})
    has_reacted = DictProperty({'like': False, 'heart': False, 'laugh': False})

    def __init__(self, **kwargs):
        super().__init__(size_hint=(0.9, None), height=dp(350), padding=dp(10), elevation=12, radius=[20], md_bg_color=[0.2, 0.2, 0.2, 1], **kwargs)
        self.mention_pattern = re.compile(r'@(\w+)')
        with self.canvas.before:
            Color(0, 0, 0, 0.4)
            Rectangle(pos=(self.x + dp(5), self.y - dp(5)), size=(self.width, self.height))
        layout = BoxLayout(orientation='vertical', spacing=dp(5))
        top_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50), padding=dp(5))
        if not os.path.exists(self.profile_image) and not self.profile_image.startswith('http'):
            self.profile_image = os.path.join(ASSETS_DIR, 'default_icon.png')
            Logger.warning(f"PostCard profile image missing, using default: {self.profile_image}")
        self.avatar = AsyncImage(source=self.profile_image, size_hint=(None, None), size=(dp(40), dp(40)))
        with self.avatar.canvas.before:
            Color(1, 1, 1, 1)
            Ellipse(pos=self.avatar.pos, size=self.avatar.size)
        top_box.add_widget(self.avatar)
        user_box = BoxLayout(orientation='vertical', padding=[dp(5), 0, 0, 0])
        user_box.add_widget(MDLabel(text=self.title, font_style="Subtitle1", theme_text_color="Custom", text_color=(1, 1, 1, 1), markup=True))
        user_box.add_widget(MDLabel(text=self.timestamp, font_style="Caption", theme_text_color="Custom", text_color=(0.7, 0.7, 0.7, 1)))
        top_box.add_widget(user_box)
        layout.add_widget(top_box)
        default_post_image = os.path.join(ASSETS_DIR, 'default_post_image.jpg')
        if not os.path.exists(default_post_image):
            Logger.warning(f"Default post image missing: {default_post_image}")
        self.image_widget = AsyncImage(source=default_post_image, size_hint_y=0.6, allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.image_widget)
        layout.add_widget(MDLabel(text=self.process_mentions(self.subtitle), font_style="Body1", theme_text_color="Custom", text_color=(0.9, 0.9, 0.9, 1), padding=[dp(10), 0, 0, 0]))
        action_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50), padding=dp(5), spacing=dp(5))
        self.like_button = MDRoundFlatButton(text=f'♥ {self.like_count}', text_color=[1, 0.2, 0.2, 1], md_bg_color=[0.3, 0.3, 0.3, 1], on_press=lambda x: self.toggle_like())
        action_box.add_widget(self.like_button)
        follow_btn = MDRoundFlatButton(text='Follow', text_color=[0, 0.5, 0, 1], md_bg_color=[0.3, 0.3, 0.3, 1], on_press=lambda x: self.toggle_follow())
        action_box.add_widget(follow_btn)
        action_box.add_widget(MDRoundFlatButton(text='💬', text_color=[0.5, 0.5, 0.5, 1], md_bg_color=[0.3, 0.3, 0.3, 1], on_press=lambda x: self.show_comments()))
        action_box.add_widget(MDRoundFlatButton(text='✉', text_color=[0.5, 0.5, 0.5, 1], md_bg_color=[0.3, 0.3, 0.3, 1], on_press=lambda x: self.send_message()))
        action_box.add_widget(MDRoundFlatButton(text='🚩', text_color=[1, 0, 0, 1], md_bg_color=[0.3, 0.3, 0.3, 1], on_press=lambda x: self.report_post()))
        layout.add_widget(action_box)
        self.add_widget(layout)
        self.bind(liked=self.update_like_state, pos=self.animate_on_touch)
        backend.increment_views(self.post_id)
        # Add AI features
        self.mood = ai_helper.analyze_mood(self.subtitle)
        if self.mood == 'positive':
            self.md_bg_color = [0.2, 0.3, 0.2, 1]
        elif self.mood == 'negative':
            self.md_bg_color = [0.3, 0.2, 0.2, 1]

    def toggle_like(self):
        try:
            app = App.get_running_app()
            if backend.toggle_like(self.post_id, app.user_id):
                self.like_count = backend.get_like_count(self.post_id)
                self.liked = not self.liked
                anim = Animation(scale=1.2, duration=0.1) + Animation(scale=1.0, duration=0.1)
                anim.start(self.like_button)
                self.update_like_state(None, self.liked)
                app.notify_new_activity()
        except Exception as e:
            Logger.error(f"PostCard toggle_like error: {e}")

    def toggle_follow(self):
        try:
            app = App.get_running_app()
            target_id = self.title.replace('[b][color=ffffff]', '').replace('[/color][/b]', '')
            backend.toggle_follow(app.user_id, target_id)
            self.children[0].children[-1].children[2].text = 'Unfollow' if 'Follow' in self.children[0].children[-1].children[2].text else 'Follow'
        except Exception as e:
            Logger.error(f"PostCard toggle_follow error: {e}")

    def update_like_state(self, instance, value):
        try:
            self.like_button.text = f'♥ {self.like_count}'
            self.like_button.text_color = [1, 0.2, 0.2, 1] if value else [0.7, 0.7, 0.7, 1]
        except Exception as e:
            Logger.error(f"PostCard update_like_state error: {e}")

    def show_comments(self):
        try:
            app = App.get_running_app()
            popup_content = BoxLayout(orientation='vertical', padding=dp(10))
            comments_list = BoxLayout(orientation='vertical', size_hint_y=0.8)
            comments = backend.get_comments(self.post_id)
            if comments:
                for comment in comments:
                    comments_list.add_widget(MDLabel(text=f"{comment['user_id']}: {comment['content']}", halign='left', theme_text_color="Custom", text_color=(0.9, 0.9, 0.9, 1)))
            else:
                comments_list.add_widget(MDLabel(text="No comments yet.", halign='center', theme_text_color="Custom", text_color=(0.7, 0.7, 0.7, 1)))
            comment_input = MDTextField(hint_text="Add a comment...", mode="rectangle")
            submit_btn = MDRoundFlatButton(text="Post", text_color=[0, 0, 0, 1], md_bg_color=[0.9, 0.5, 0.5, 1], pos_hint={'center_x': 0.5})
            popup_content.add_widget(comments_list)
            popup_content.add_widget(comment_input)
            popup_content.add_widget(submit_btn)
            popup = Popup(title=f'Comments on {self.post_id}', content=popup_content, size_hint=(0.8, 0.7), background_color=[0, 0, 0, 0.9])
            submit_btn.bind(on_press=lambda x: self.add_comment(comment_input.text, comments_list, popup))
            popup.open()
        except Exception as e:
            Logger.error(f"PostCard show_comments error: {e}")

    def add_comment(self, text, comments_list, popup):
        try:
            app = App.get_running_app()
            if text.strip() and backend.add_comment(self.post_id, app.user_id, text):
                comments_list.clear_widgets()
                for comment in backend.get_comments(self.post_id):
                    comments_list.add_widget(MDLabel(text=f"{comment['user_id']}: {comment['content']}", halign='left', theme_text_color="Custom", text_color=(0.9, 0.9, 0.9, 1)))
                popup.dismiss()
        except Exception as e:
            Logger.error(f"PostCard add_comment error: {e}")

    def send_message(self):
        try:
            app = App.get_running_app()
            target_id = self.title.replace('[b][color=ffffff]', '').replace('[/color][/b]', '')
            popup = Popup(title=f'Message {target_id}', size_hint=(0.8, 0.8))
        except Exception as e:
            Logger.error(f"PostCard submit_message error: {e}")

    def report_post(self):
        try:
            app = App.get_running_app()
            popup = Popup(title='Report Post', size_hint=(0.8, 0.5))
            layout = BoxLayout(orientation='vertical', padding=10)
            reason_input = MDTextField(hint_text="Reason for reporting...", mode="rectangle", multiline=True)
            submit_btn = MDRoundFlatButton(text='Report', pos_hint={'center_x': 0.5}, on_press=lambda x: self.submit_report(reason_input.text, popup))
            layout.add_widget(reason_input)
            layout.add_widget(submit_btn)
            popup.content = layout
            popup.open()
        except Exception as e:
            Logger.error(f"PostCard report_post error: {e}")

    def submit_report(self, reason, popup):
        try:
            app = App.get_running_app()
            if reason.strip() and backend.report_post(self.post_id, app.user_id, reason):
                popup.dismiss()
        except Exception as e:
            Logger.error(f"PostCard submit_report error: {e}")

    def animate_on_touch(self, instance, pos):
        try:
            if self.collide_point(*self.to_widget(*pos)):
                anim = Animation(elevation=16, duration=0.1) + Animation(elevation=12, duration=0.1)
                anim.start(self)
        except Exception as e:
            Logger.error(f"PostCard animate_on_touch error: {e}")

    def process_mentions(self, text):
        mentions = self.mention_pattern.findall(text)
        for mention in mentions:
            text = text.replace(f"@{mention}", f"[ref={mention}]@{mention}[/ref]")
        return text

    def on_ref_press(self, ref):
        app = App.get_running_app()
        profile = cache_manager.get_profile(ref)
        if profile:
            self.show_profile(ref)

    def toggle_reaction(self, reaction_type):
        try:
            self.has_reacted[reaction_type] = not self.has_reacted[reaction_type]
            self.reactions[reaction_type] += 1 if self.has_reacted[reaction_type] else -1
            # Animate reaction button
            reaction_btn = self.ids.reaction_box.children[reaction_type]
            anim = (Animation(size=(dp(45), dp(45)), duration=0.1) + 
                   Animation(size=(dp(40), dp(40)), duration=0.1))
            anim.start(reaction_btn)
        except Exception as e:
            Logger.error(f"toggle_reaction error: {e}")

    def share_post(self, *args):
        if ai_helper.moderate_content(self.subtitle, self.image_url):
            super().share_post(*args)
        else:
            toast("Content flagged by AI moderation")

class LoginScreen(Popup):
    status = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(title='Login / Register', size_hint=(0.8, 0.8), **kwargs)
        self.status = ""
        layout = BoxLayout(orientation='vertical', padding=10)
        self.username = MDTextField(hint_text="Username", mode="rectangle")
        self.password = MDTextField(hint_text="Password", mode="rectangle", password=True)
        self.status_label = MDLabel(text=self.status, halign='center', theme_text_color="Error")
        layout.add_widget(self.username)
        layout.add_widget(self.password)
        layout.add_widget(self.status_label)
        layout.add_widget(MDRoundFlatButton(text='Login', pos_hint={'center_x': 0.5}, on_press=lambda x: self.login()))
        layout.add_widget(MDRoundFlatButton(text='Register', pos_hint={'center_x': 0.5}, on_press=lambda x: self.register()))
        self.content = layout

    def login(self):
        try:
            app = App.get_running_app()
            if self.username.text and self.password.text:
                if backend.login_user(self.username.text, self.password.text):
                    app.logged_in = True
                    app.user_id = self.username.text
                    app.store.put('user', logged_in=True, user_id=self.username.text)
                    # Fix: Handle case when 'posts' key doesn't exist
                    if app.store.exists('posts'):
                        local_posts = app.store.get('posts')['local_posts']
                        for feature, content in local_posts:
                            backend.share_post(feature, content, app.user_id)
                    app.store.put('posts', local_posts=[])
                    self.dismiss()
                else:
                    self.status = "Invalid username or password"
            else:
                self.status = "Please enter username and password"
            self.status_label.text = self.status
        except Exception as e:
            Logger.error(f"LoginScreen login error: {e}")

    def register(self):
        try:
            app = App.get_running_app()
            if self.username.text and self.password.text:
                if backend.register_user(self.username.text, self.password.text):
                    app.logged_in = True
                    app.user_id = self.username.text
                    app.store.put('user', logged_in=True, user_id=self.username.text)
                    self.dismiss()
                else:
                    self.status = "Username already taken"
            else:
                self.status = "Please enter username and password"
            self.status_label.text = self.status
        except Exception as e:
            Logger.error(f"LoginScreen register error: {e}")

    def dismiss(self):
        try:
            self.status = ""
            self.status_label.text = ""
            super().dismiss()
        except Exception as e:
            Logger.error(f"LoginScreen dismiss error: {e}")

class HomeScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stories = {}
        Clock.schedule_once(self.load_stories, 0)

    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('vibe', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'feed'):
                Logger.error("HomeScreen: feed not found in ids")
                return
            self.ids.feed.clear_widgets()
            app = App.get_running_app()
            all_posts = []
            features = ['vibe', 'echo', 'frame', 'pulse', 'capsule', 'roots', 'spark', 'glow', 'challenge']
            for feature in features:
                all_posts.extend([{"user_id": p['user_id'], "content": p['content'], "timestamp": self.format_time(p["timestamp"]), "id": p['id'], "image_url": p['image_url'], "collaborators": p['collaborators']}
                                 for p in backend.get_posts(feature, limit=self.limit, offset=self.offset)])
            orbits = ['Family', 'Friends', 'Work', 'Hobbies']
            for orbit in orbits:
                all_posts.extend([{"user_id": p['user_id'], "content": p['content'], "orbit": orbit, "timestamp": self.format_time(p["timestamp"]), "id": p['id'], "image_url": p['image_url'], "collaborators": p['collaborators']}
                                 for p in backend.get_posts('orbit', orbit, limit=self.limit, offset=self.offset)])
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            all_posts.extend([{"user_id": "You", "content": content, "timestamp": "Just now", "id": f"local_{len(all_posts)}", "image_url": "", "collaborators": []}
                             for _, content in local_posts])
            for post in all_posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = f"{post['orbit']}: {post['content']}" if 'orbit' in post else post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = post['timestamp']
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.feed.add_widget(card)
            self.ids.feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"HomeScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        try:
            delta = datetime.now() - datetime.fromtimestamp(timestamp)
            if delta.days > 1:
                return f"{datetime.fromtimestamp(timestamp).strftime('%b %d')}"
            elif delta.days == 1:
                return "Yesterday"
            elif delta.seconds < 60:
                return "Just now"
            elif delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"{minutes}m"
            else:
                hours = delta.seconds // 3600
                return f"{hours}h"
        except Exception as e:
            Logger.error(f"HomeScreen format_time error: {e}")
            return "Unknown time"

    def load_stories(self, dt):
        try:
            # Get stories from backend
            stories = backend.get_stories(App.get_running_app().user_id)
            for story in stories:
                circle = StoryCircle(profile_image=story.media_url)
                self.ids.story_container.add_widget(circle)
        except Exception as e:
            Logger.error(f"load_stories error: {e}")

    def view_story(self, story_circle):
        try:
            # Show story viewer popup
            popup = Popup(title='Story', size_hint=(0.9, 0.9))
            content = AsyncImage(source=story_circle.profile_image)
            popup.content = content
            popup.open()
            # Start story timer
            story_circle.start_animation()
            Clock.schedule_once(lambda dt: popup.dismiss(), story_circle.duration)
        except Exception as e:
            Logger.error(f"view_story error: {e}")

class VibeSphereScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('vibe', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'vibe_feed'):
                Logger.error("VibeSphereScreen: vibe_feed not found in ids")
                return
            self.ids.vibe_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('vibe', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'vibe'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.vibe_feed.add_widget(card)
            self.ids.vibe_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"VibeSphereScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def share_vibe(self, vibe):
        try:
            app = App.get_running_app()
            if vibe.strip():
                if app.logged_in:
                    post_id = backend.share_post('vibe', vibe, app.user_id)
                    if post_id:
                        self.update_feed()
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('vibe', vibe))
                    app.store.put('posts', local_posts=local_posts)
                    self.update_feed()
        except Exception as e:
            Logger.error(f"VibeSphereScreen share_vibe error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class EchoScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recording = False
        self.audio_data = []
        self.fs = 44100

    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('echo', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'echo_list'):
                Logger.error("EchoScreen: echo_list not found in ids")
                return
            self.ids.echo_list.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('echo', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'echo'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = f"Echo: {post['content']}"
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.echo_list.add_widget(card)
            self.ids.echo_list.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"EchoScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.audio_data = []
            self.ids.record_button.text = "Recording..."
            Clock.schedule_once(self._record, 0)

    def _record(self, dt):
        try:
            audio = sd.rec(int(10 * self.fs), samplerate=self.fs, channels=1, dtype='float32')
            sd.wait()
            self.audio_data = audio
            if self.recording:
                self.stop_recording()
        except Exception as e:
            Logger.error(f"EchoScreen _record error: {e}")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.ids.record_button.text = "Record"
            if self.audio_data:
                filename = os.path.join(ASSETS_DIR, f"echo_{int(time())}.wav")
                sf.write(filename, self.audio_data, self.fs)
                self.create_soundscape(filename)

    def create_soundscape(self, filename):
        try:
            data, fs = sf.read(filename)
            amplified = (data * 1.5).clip(-1, 1)
            echo = np.concatenate([np.zeros(int(0.5 * fs)), amplified[:-int(0.5 * fs)]])
            soundscape = (amplified + echo * 0.5).clip(-1, 1)
            soundscape_file = os.path.join(ASSETS_DIR, f"soundscape_{int(time())}.wav")
            sf.write(soundscape_file, soundscape, fs)
            blob = bucket.blob(f'audio/{os.path.basename(soundscape_file)}')
            blob.upload_from_filename(soundscape_file)
            blob.make_public()
            audio_url = blob.public_url
            app = App.get_running_app()
            if app.logged_in:
                backend.share_post('echo', f"Soundscape created", app.user_id, audio_url)
            else:
                local_posts = app.store.get('posts')['local_posts']
                local_posts.append(('echo', audio_url))
                app.store.put('posts', local_posts=local_posts)
            self.update_feed()
        except Exception as e:
            Logger.error(f"EchoScreen create_soundscape error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class FrameScreen(BaseScreen):
    timelapse_speed = NumericProperty(2.0)

    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('frame', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'frame_feed'):
                Logger.error("FrameScreen: frame_feed not found in ids")
                return
            self.ids.frame_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('frame', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'frame'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = "Timelapse Frame"
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.frame_feed.add_widget(card)
            self.ids.frame_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"FrameScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def open_frame_maker(self):
        try:
            popup = Popup(title='Create Timelapse Frame', size_hint=(0.8, 0.8))
            layout = BoxLayout(orientation='vertical', padding=10)
            file_chooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg'])
            collab_input = MDTextField(hint_text="Collaborators (comma-separated usernames)", mode="rectangle")
            layout.add_widget(file_chooser)
            layout.add_widget(collab_input)
            layout.add_widget(MDRoundFlatButton(text='Add Frames', pos_hint={'center_x': 0.5}, on_press=lambda x: self.share_frame(file_chooser.selection, collab_input.text, popup)))
            popup.content = layout
            popup.open()
        except Exception as e:
            Logger.error(f"FrameScreen open_frame_maker error: {e}")

    def share_frame(self, files, collab_text, popup):
        try:
            if files:
                app = App.get_running_app()
                image_urls = []
                for file in files:
                    blob = bucket.blob(f'frames/{app.user_id}/{os.path.basename(file)}')
                    blob.upload_from_filename(file)
                    blob.make_public()
                    image_urls.append(blob.public_url)
                content = ';'.join(image_urls) + ';'
                collaborators = [c.strip() for c in collab_text.split(',') if c.strip()] if collab_text else []
                if app.logged_in:
                    backend.share_post('frame', content, app.user_id, image_urls[0], collaborators)
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('frame', content))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
                popup.dismiss()
        except Exception as e:
            Logger.error(f"FrameScreen share_frame error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class PulseScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('pulse', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'pulse_feed'):
                Logger.error("PulseScreen: pulse_feed not found in ids")
                return
            self.ids.pulse_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('pulse', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'pulse'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.pulse_feed.add_widget(card)
            self.ids.pulse_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"PulseScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def share_pulse(self, pulse):
        try:
            app = App.get_running_app()
            if pulse.strip():
                if app.logged_in:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('pulse', pulse))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
        except Exception as e:
            Logger.error(f"PulseScreen share_pulse error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class TimeCapsuleScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('capsule', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'capsule_feed'):
                Logger.error("TimeCapsuleScreen: capsule_feed not found in ids")
                return
            self.ids.capsule_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('capsule', limit=self.limit, offset=self.offset)
            capsules = backend.get_capsules(app.user_id)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'capsule'])
            for capsule in capsules:
                if not capsule['unlocked']:
                    posts.append({"id": capsule['id'], "user_id": app.user_id, "content": f"{capsule['content']} (Unlocks in {(capsule['unlock_time'] - time()) / 86400:.1f} days)", "timestamp": capsule['unlock_time'], "image_url": "", "collaborators": []})
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.capsule_feed.add_widget(card)
            self.ids.capsule_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"TimeCapsuleScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def share_capsule(self, capsule):
        try:
            app = App.get_running_app()
            if capsule.strip() and 'days_spinner' in self.ids:
                days = int(self.ids.days_spinner.text)
                unlock_time = (datetime.now() + timedelta(days=days)).timestamp()
                if app.logged_in:
                    backend.schedule_capsule(app.user_id, capsule, unlock_time)
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('capsule', f"{capsule} (Unlocks in {days} days)"))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
        except Exception as e:
            Logger.error(f"TimeCapsuleScreen share_capsule error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class RootsScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('roots', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'roots_feed'):
                Logger.error("RootsScreen: roots_feed not found in ids")
                return
            self.ids.roots_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('roots', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'roots'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.roots_feed.add_widget(card)
            self.ids.roots_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"RootsScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def share_roots(self, roots):
        try:
            app = App.get_running_app()
            if roots.strip():
                if app.logged_in:
                    backend.share_post('roots', roots, app.user_id)
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('roots', roots))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
        except Exception as e:
            Logger.error(f"RootsScreen share_roots error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class SparkScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('spark', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'spark_feed'):
                Logger.error("SparkScreen: spark_feed not found in ids")
                return
            self.ids.spark_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('spark', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'spark'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = post['content'].split(': ')[0] if ': ' in post['content'] else post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.spark_feed.add_widget(card)
            self.ids.spark_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"SparkScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def open_spark_maker(self):
        try:
            popup = Popup(title='Create Spark', size_hint=(0.8, 0.8))
            layout = BoxLayout(orientation='vertical', padding=10)
            text_input = MDTextField(hint_text="Write your spark...", mode="rectangle", multiline=True)
            file_chooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg'])
            collab_input = MDTextField(hint_text="Collaborators (comma-separated usernames)", mode="rectangle")
            layout.add_widget(text_input)
            layout.add_widget(file_chooser)
            layout.add_widget(collab_input)
            layout.add_widget(MDRoundFlatButton(text='Edit Image', pos_hint={'center_x': 0.5}, on_press=lambda x: self.edit_image(file_chooser.selection, popup)))
            layout.add_widget(MDRoundFlatButton(text='Share', pos_hint={'center_x': 0.5}, on_press=lambda x: self.share_spark(text_input.text, file_chooser.selection, collab_input.text, popup)))
            popup.content = layout
            popup.open()
        except Exception as e:
            Logger.error(f"SparkScreen open_spark_maker error: {e}")

    def edit_image(self, files, parent_popup):
        try:
            if not files:
                return
            popup = Popup(title='Edit Image', size_hint=(0.8, 0.8))
            layout = BoxLayout(orientation='vertical', padding=10)
            image = DraggableImage(source=files[0], size_hint=(1, 0.7))
            brightness_slider = Builder.load_string('Slider:\n    min: 0\n    max: 2\n    value: 1\n    hint: False')
            layout.add_widget(image)
            layout.add_widget(MDLabel(text="Brightness"))
            layout.add_widget(brightness_slider)
            layout.add_widget(MDRoundFlatButton(text='Save', pos_hint={'center_x': 0.5}, on_press=lambda x: self.save_edited_image(files[0], brightness_slider.value, popup, parent_popup)))
            brightness_slider.bind(value=lambda instance, value: self.adjust_brightness(image, value))
            popup.content = layout
            popup.open()
        except Exception as e:
            Logger.error(f"SparkScreen edit_image error: {e}")

    def adjust_brightness(self, image_widget, value):
        try:
            pil_image = PILImage.open(image_widget.source)
            pil_image = pil_image.point(lambda p: p * value)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_widget.source = ''
            image_widget.reload()
            image_widget.texture = CoreImage(buffer, ext='png').texture
        except Exception as e:
            Logger.error(f"SparkScreen adjust_brightness error: {e}")

    def save_edited_image(self, original_path, brightness, popup, parent_popup):
        try:
            pil_image = PILImage.open(original_path)
            pil_image = pil_image.point(lambda p: p * brightness)
            edited_path = os.path.join(ASSETS_DIR, f"edited_{int(time())}.png")
            pil_image.save(edited_path)
            popup.dismiss()
            parent_popup.content.children[-2].selection = [edited_path]  # Update file chooser selection
        except Exception as e:
            Logger.error(f"SparkScreen save_edited_image error: {e}")

    def share_spark(self, text, files, collab_text, popup):
        try:
            app = App.get_running_app()
            if text.strip():
                image_url = None
                if files:
                    blob = bucket.blob(f'sparks/{app.user_id}/{os.path.basename(files[0])}')
                    blob.upload_from_filename(files[0])
                    blob.make_public()
                    image_url = blob.public_url
                collaborators = [c.strip() for c in collab_text.split(',') if c.strip()] if collab_text else []
                if app.logged_in:
                    backend.share_post('spark', text, app.user_id, image_url, collaborators)
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('spark', text))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
                popup.dismiss()
        except Exception as e:
            Logger.error(f"SparkScreen share_spark error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

    def use_recommendation(self, post):
        try:
            # Use the recommended post as a template
            if hasattr(self.ids, 'text_input'):
                self.ids.text_input.text = post['content']
        except Exception as e:
            Logger.error(f"SparkScreen use_recommendation error: {e}")

class GlowScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('glow', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'glow_feed'):
                Logger.error("GlowScreen: glow_feed not found in ids")
                return
            self.ids.glow_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('glow', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'glow'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.glow_feed.add_widget(card)
            self.ids.glow_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"GlowScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def share_glow(self, glow):
        try:
            app = App.get_running_app()
            if glow.strip():
                if app.logged_in:
                    backend.share_post('glow', glow, app.user_id)
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('glow', glow))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
        except Exception as e:
            Logger.error(f"GlowScreen share_glow error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class ChallengeScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        self.listeners.append(backend.listen_to_posts('challenge', self.update_feed_realtime))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'challenge_feed'):
                Logger.error("ChallengeScreen: challenge_feed not found in ids")
                return
            self.ids.challenge_feed.clear_widgets()
            app = App.get_running_app()
            posts = backend.get_posts('challenge', limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content, "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (_, content) in enumerate(local_posts) if _ == 'challenge'])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = post['content']
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.challenge_feed.add_widget(card)
            self.ids.challenge_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"ChallengeScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def share_challenge(self, challenge):
        try:
            app = App.get_running_app()
            if challenge.strip():
                if app.logged_in:
                    backend.share_post('challenge', challenge, app.user_id)
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('challenge', challenge))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
        except Exception as e:
            Logger.error(f"ChallengeScreen share_challenge error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class OrbitScreen(BaseScreen):
    def on_pre_enter(self):
        self.offset = 0
        self.update_feed()
        for orbit in ['Family', 'Friends', 'Work', 'Hobbies']:
            self.listeners.append(backend.listen_to_posts('orbit', self.update_feed_realtime, orbit))

    def update_feed(self):
        try:
            if not hasattr(self.ids, 'orbit_feed'):
                Logger.error("OrbitScreen: orbit_feed not found in ids")
                return
            self.ids.orbit_feed.clear_widgets()
            app = App.get_running_app()
            orbit = self.ids.orbit_spinner.text if 'orbit_spinner' in self.ids else 'Family'
            posts = backend.get_posts('orbit', orbit, limit=self.limit, offset=self.offset)
            local_posts = app.store.get('posts')['local_posts'] if not app.logged_in else []
            posts.extend([{"id": f"local_{i}", "user_id": "You", "content": content.split(':')[1], "timestamp": time(), "image_url": "", "collaborators": []}
                          for i, (feat, content) in enumerate(local_posts) if feat == 'orbit' and content.startswith(orbit)])
            for post in posts:
                if app.logged_in and backend.get_profile(post['user_id'])['privacy'] == 'private' and app.user_id != post['user_id'] and app.user_id not in backend.db.collection('follows').where('following_id', '==', post['user_id']).where('follower_id', '==', app.user_id).get():
                    continue
                card = PostCard()
                card.opacity = 0
                anim = Animation(opacity=1, duration=0.5, t='out_quad')
                anim.start(card)
                card.title = "[b][color=ffffff]You[/color][/b]" if post['user_id'] == "You" else post['user_id']
                card.subtitle = f"{orbit}: {post['content']}"
                if post['collaborators']:
                    card.subtitle += f" (with {', '.join(post['collaborators'])})"
                card.timestamp = self.format_time(post['timestamp'])
                profile = backend.get_profile(post['user_id'])
                card.profile_image = profile['profile_pic'] if profile and profile['profile_pic'] else os.path.join(ASSETS_DIR, 'default_icon.png')
                card.post_id = post['id']
                card.like_count = backend.get_like_count(post['id'])
                card.image_url = post['image_url']
                if post['image_url']:
                    card.image_widget.source = post['image_url']
                self.ids.orbit_feed.add_widget(card)
            self.ids.orbit_feed.add_widget(MDRoundFlatButton(text="Load More", pos_hint={'center_x': 0.5}, on_press=lambda x: self.load_more()))
        except Exception as e:
            Logger.error(f"OrbitScreen update_feed error: {e}")

    def update_feed_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_feed())

    def share_orbit(self, orbit_text):
        try:
            app = App.get_running_app()
            if orbit_text.strip() and 'orbit_spinner' in self.ids:
                orbit = self.ids.orbit_spinner.text
                content = f"{orbit}:{orbit_text}"
                if app.logged_in:
                    backend.share_post('orbit', content, app.user_id)
                else:
                    local_posts = app.store.get('posts')['local_posts']
                    local_posts.append(('orbit', content))
                    app.store.put('posts', local_posts=local_posts)
                self.update_feed()
        except Exception as e:
            Logger.error(f"OrbitScreen share_orbit error: {e}")

    def load_more(self):
        self.offset += self.limit
        self.update_feed()

    def format_time(self, timestamp):
        return HomeScreen().format_time(timestamp)

class MessagesScreen(BaseScreen):
    selected_user = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voice_note = VoiceNote()
        
    def send_voice_note(self):
        if self.voice_note.recording:
            filename = self.voice_note.stop_recording()
            if filename:
                self.upload_and_send_voice_note(filename)
        else:
            self.voice_note.start_recording()
            
    def upload_and_send_voice_note(self, filename):
        try:
            blob = bucket.blob(f'voice_notes/{os.path.basename(filename)}')
            blob.upload_from_filename(filename)
            blob.make_public()
            url = blob.public_url
            
            app = App.get_running_app()
            if self.selected_user:
                backend.send_message(app.user_id, self.selected_user, f"[voice_note]{url}")
                self.update_messages()
        except Exception as e:
            Logger.error(f"Error uploading voice note: {e}")

    def on_pre_enter(self):
        self.update_user_list()
        self.listeners.append(backend.listen_to_messages(App.get_running_app().user_id, self.update_messages_realtime))

    def update_user_list(self):
        try:
            if not hasattr(self.ids, 'user_list'):
                Logger.error("MessagesScreen: user_list not found in ids")
                return
            self.ids.user_list.clear_widgets()
            app = App.get_running_app()
            users = set()
            messages = backend.get_messages(app.user_id, "")
            for msg in messages:
                users.add(msg['sender_id'] if msg['sender_id'] != app.user_id else msg['receiver_id'])
            for user in users:
                btn = MDRoundFlatButton(text=user, pos_hint={'center_x': 0.5}, on_press=lambda x, u=user: self.select_user(u))
                self.ids.user_list.add_widget(btn)
        except Exception as e:
            Logger.error(f"MessagesScreen update_user_list error: {e}")

    def select_user(self, user):
        self.selected_user = user
        self.update_messages()

    def update_messages(self):
        try:
            if not hasattr(self.ids, 'message_list'):
                Logger.error("MessagesScreen: message_list not found in ids")
                return
            self.ids.message_list.clear_widgets()
            app = App.get_running_app()
            messages = backend.get_messages(app.user_id, self.selected_user)
            for msg in messages:
                sender = "You" if msg['sender_id'] == app.user_id else msg['sender_id']
                if msg['content'].startswith('[voice_note]'):
                    # Add voice note player for audio messages
                    url = msg['content'].replace('[voice_note]', '')
                    player = VoiceNotePlayer(url=url)
                    self.ids.message_list.add_widget(player)
                else:
                    # Regular text message
                    self.ids.message_list.add_widget(MDLabel(
                        text=f"{sender}: {msg['content']}", 
                        halign='left' if sender == "You" else 'right',
                        theme_text_color="Custom",
                        text_color=(1, 1, 1, 1)
                    ))
        except Exception as e:
            Logger.error(f"MessagesScreen update_messages error: {e}")

    def update_messages_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_messages())

    def send_message(self, message):
        try:
            app = App.get_running_app()
            if message.strip() and self.selected_user:
                backend.send_message(app.user_id, self.selected_user, message)
                self.update_messages()
        except Exception as e:
            Logger.error(f"MessagesScreen send_message error: {e}")

class NotificationsScreen(BaseScreen):
    def on_pre_enter(self):
        self.update_notifications()
        self.listeners.append(backend.listen_to_notifications(App.get_running_app().user_id, self.update_notifications_realtime))

    def update_notifications(self):
        try:
            if not hasattr(self.ids, 'notification_list'):
                Logger.error("NotificationsScreen: notification_list not found in ids")
                return
            self.ids.notification_list.clear_widgets()
            app = App.get_running_app()
            notifications = backend.get_notifications(app.user_id)
            for notif in notifications:
                label = MDLabel(text=notif['content'], halign='left', theme_text_color="Custom", text_color=(1, 1, 1, 1) if not notif['read'] else (0.7, 0.7, 0.7, 1))
                label.bind(on_touch_down=lambda instance, touch, n=notif: self.mark_read(n['timestamp']) if instance.collide_point(*touch.pos) else None)
                self.ids.notification_list.add_widget(label)
        except Exception as e:
            Logger.error(f"NotificationsScreen update_notifications error: {e}")

    def update_notifications_realtime(self, docs):
        Clock.schedule_once(lambda dt: self.update_notifications())

    def mark_read(self, timestamp):
        try:
            app = App.get_running_app()
            backend.mark_notification_read(app.user_id, timestamp)
            self.update_notifications()
        except Exception as e:
            Logger.error(f"NotificationsScreen mark_read error: {e}")

class ConnectSphereApp(MDApp):
    logged_in = BooleanProperty(False)
    user_id = StringProperty("Guest")
    store = ObjectProperty()
    progress = NumericProperty(0)  # Add missing property

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.store = JsonStore(os.path.join(BASE_DIR, 'local_storage.json'))
        if not self.store.exists('posts'):
            self.store.put('posts', local_posts=[])
        if self.store.exists('user'):
            user_data = self.store.get('user')
            self.logged_in = user_data['logged_in']
            self.user_id = user_data['user_id']
        # Initialize AR support
        self.ar_enabled = False
        self.vr_mode = False

    def build(self):
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.theme_style = "Dark"
        Builder.load_file(os.path.join(BASE_DIR, 'connectsphere.kv'))
        return Builder.load_string('ScreenManager:\n    HomeScreen:\n    VibeSphereScreen:\n    EchoScreen:\n    FrameScreen:\n    PulseScreen:\n    TimeCapsuleScreen:\n    RootsScreen:\n    SparkScreen:\n    GlowScreen:\n    ChallengeScreen:\n    OrbitScreen:\n    MessagesScreen:\n    NotificationsScreen:')

    def notify_new_activity(self):
        Clock.schedule_once(lambda dt: NotificationsScreen().update_notifications())

    def toggle_vr_mode(self):
        try:
            self.vr_mode = not self.vr_mode
            if self.vr_mode:
                Window.fullscreen = True
                self.root.canvas.before.clear()
                with self.root.canvas.before:
                    Color(1, 1, 1, 1)
                    self.root.vr_texture = Rectangle(size=self.root.size)
            else:
                Window.fullscreen = False
                self.root.canvas.before.clear()
        except Exception as e:
            Logger.error(f"VR mode toggle error: {e}")
            self.vr_mode = False
            Window.fullscreen = False
            if hasattr(self.root, 'canvas'):
                self.root.canvas.before.clear()

class ReactionButton(ButtonBehavior, BoxLayout):
    icon = StringProperty()
    icon_color = ListProperty([0.7, 0.7, 0.7, 1])
    count = NumericProperty(0)
    active = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.spacing = dp(5)
        self.bind(on_press=self.toggle_reaction)
        
        with self.canvas.before:
            Color(*self.icon_color)
            self.bg = Ellipse(pos=self.pos, size=self.size)
            
        self.icon_label = MDLabel(
            text=self.icon,
            halign='center',
            theme_text_color="Custom",
            text_color=self.icon_color
        )
        self.count_label = MDLabel(
            text=str(self.count),
            halign='left',
            theme_text_color="Custom",
            text_color=self.icon_color
        )
        
        self.add_widget(self.icon_label)
        self.add_widget(self.count_label)

    def toggle_reaction(self, *args):
        self.active = not self.active
        self.count += 1 if self.active else -1
        
        # Animate the button
        scale_up = Animation(scale=1.2, duration=0.1)
        scale_down = Animation(scale=1.0, duration=0.1)
        color_change = Animation(icon_color=[1, 0.8, 0, 1] if self.active else [0.7, 0.7, 0.7, 1], 
                               duration=0.2)
        
        anim = scale_up + scale_down & color_change
        anim.start(self)

class OrbitConstellationScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stars = []
        self.connections = []
        self.selected_star = None
        
    def draw_constellation(self):
        try:
            self.canvas.clear()
            with self.canvas:
                Color(1, 1, 1, 0.8)
                app = App.get_running_app()
                orbits = ['Family', 'Friends', 'Work', 'Hobbies']
                y_offset = dp(100)
                radius = dp(5)
                
                for orbit in orbits:
                    posts = backend.get_posts('orbit', orbit)
                    x_offset = dp(100)
                    
                    for post in posts:
                        # Draw star with glow effect
                        Color(0.5, 0.5, 1, 0.3)
                        glow = Ellipse(pos=(x_offset-radius, y_offset-radius), 
                                     size=(radius*4, radius*4))
                        
                        Color(1, 1, 1, 0.8)
                        star = Ellipse(pos=(x_offset, y_offset), 
                                     size=(radius*2, radius*2))
                        
                        self.stars.append({
                            'user_id': post['user_id'],
                            'star': star,
                            'glow': glow,
                            'post': post,
                            'pos': (x_offset+radius, y_offset+radius)
                        })
                        
        except Exception as e:
            Logger.error(f"OrbitConstellationScreen draw_constellation error: {e}")

    def update_connections(self):
        if not self.selected_star:
            return
            
        with self.canvas:
            Color(0.5, 0.5, 1, 0.4)
            for star in self.stars:
                if (star['user_id'] == self.selected_star['user_id'] and 
                    star != self.selected_star):
                    points = [
                        self.selected_star['pos'][0],
                        self.selected_star['pos'][1],
                        star['pos'][0],
                        star['pos'][1]
                    ]
                    line = Line(points=points, width=dp(1))
                    self.connections.append(line)

    def on_touch_down(self, touch):
        for star in self.stars:
            if star['star'].collide_point(*touch.pos):
                self.selected_star = star
                self.show_star_details(star)
                self.update_connections()
                return True
        return super().on_touch_down(touch)

    def show_star_details(self, star):
        popup = Popup(
            title=f"Post by {star['user_id']}",
            size_hint=(0.8, 0.4),
            background_color=[0.1, 0.1, 0.2, 0.95]
        )
        content = BoxLayout(orientation='vertical', padding=dp(10))
        content.add_widget(MDLabel(
            text=star['post']['content'],
            theme_text_color="Custom",
            text_color=[1, 1, 1, 0.9]
        ))
        popup.content = content
        popup.open()

class StoryViewer(Popup):
    current_story = ObjectProperty(None)
    is_playing = BooleanProperty(False)
    current_position = NumericProperty(0)
    music_duration = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_music = None
        self.drawing_enabled = False
        self._touches = []
        self.music_player = None
        
    def toggle_audio(self):
        if self.current_story and hasattr(self.current_story, 'audio'):
            if self.current_story.audio.state == 'play':
                self.current_story.audio.stop()
            else:
                self.current_story.audio.play()
                
    def select_background_music(self):
        file_chooser = FileChooserIconView(filters=['*.mp3'])
        popup = Popup(title='Select Music', content=file_chooser)
        file_chooser.bind(on_submit=lambda x: self.set_background_music(x.selection[0]))
        popup.open()
        
    def start_drawing(self):
        self.drawing_enabled = True
        
    def on_touch_down(self, touch):
        if self.drawing_enabled:
            self._touches.append(touch)
            with self.canvas:
                Color(1, 1, 1, 0.8)
                touch.ud['line'] = Line(points=(touch.x, touch.y))
        return super().on_touch_down(touch)

    def format_time(self, seconds):
        try:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"
        except Exception as e:
            Logger.error(f"StoryViewer format_time error: {e}")
            return "00:00"
        
    def toggle_playback(self):
        if self.music_player:
            if self.is_playing:
                self.music_player.pause()
            else:
                self.music_player.play()
            self.is_playing = not self.is_playing
            
    def seek_music(self, value):
        if self.music_player:
            self.music_player.seek(value)
            self.current_position = value

    def on_touch_move(self, touch):
        if self.drawing_enabled and touch in self._touches:
            touch.ud['line'].points += [touch.x, touch.y]
        return super().on_touch_move(touch)
            
    def on_touch_up(self, touch):
        if self.drawing_enabled and touch in self._touches:
            self._touches.remove(touch)
        return super().on_touch_up(touch)

class CollaborativeDrawing:
    def __init__(self):
        self.session_id = None
        self.participants = set()
        self.strokes = []
        
    def start_session(self, user_id):
        self.session_id = f"drawing_{int(time())}"
        self.participants.add(user_id)
        
    def add_stroke(self, points, color, width):
        stroke = {
            'points': points,
            'color': color,
            'width': width,
            'timestamp': time()
        }
        self.strokes.append(stroke)
        self.broadcast_stroke(stroke)
        
    def broadcast_stroke(self, stroke):
        for participant in self.participants:
            backend.send_drawing_update(self.session_id, participant, stroke)

class VoiceNote:
    def __init__(self):
        super().__init__()
        self.playing = False
        self.recording = False
        self.audio_data = []
        self.sample_rate = 44100
        
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        threading.Thread(target=self._record).start()
        
    def stop_recording(self):
        self.recording = False
        return self.save_recording()
        
    def _record(self):
        with sd.InputStream(callback=self._audio_callback,
                          channels=1,
                          samplerate=self.sample_rate):
            while self.recording:
                sd.sleep(100)
                
    def _audio_callback(self, indata, frames, time, status):
        self.audio_data.append(indata.copy())
        
    def save_recording(self):
        if not self.audio_data:
            return None
        filename = os.path.join(ASSETS_DIR, f"voice_{int(time())}.wav")
        data = np.concatenate(self.audio_data)
        sf.write(filename, data, self.sample_rate)
        return filename

class VoiceNotePlayer(BoxLayout):
    url = StringProperty("")
    playing = BooleanProperty(False)
    duration = NumericProperty(0)
    position = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sound = None
        if self.url:
            self.sound = SoundLoader.load(self.url)
            if self.sound:
                self.duration = self.sound.length
    
    def toggle_play(self):
        if not self.sound:
            return
            
        if self.playing:
            self.sound.stop()
        else:
            self.sound.play()
        self.playing = not self.playing
        
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

class StoryCircle(ButtonBehavior, BoxLayout):
    profile_image = StringProperty("")
    duration = NumericProperty(5)
    progress = NumericProperty(0)
    
    def start_animation(self):
        anim = Animation(progress=100, duration=self.duration)
        anim.start(self)

class CollaborativeDrawingPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Collaborative Drawing"
        self.size_hint = (0.9, 0.9)
        
    def save_drawing(self):
        # Save drawing logic
        texture = self.export_as_image()
        # Save to file or upload
        self.dismiss()
