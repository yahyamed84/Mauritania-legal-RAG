import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-placeholder-key-for-dev'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'channels',
    'rag_app',
    'whitenoise.runserver_nostatic',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'mauritania_rag.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'mauritania_rag.wsgi.application'
ASGI_APPLICATION = 'mauritania_rag.asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'ar'
TIME_ZONE = 'Africa/Nouakchott'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REDIS_URL = 'redis://redis:6379/0'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {'hosts': [REDIS_URL]},
    },
}

# OPTIMIZED RAG Configuration for Maximum Speed
RAG_CONFIG = {
    # Paths
    'DATA_DIR': BASE_DIR / 'data',
    'TEXT_DIR': BASE_DIR / 'data' / 'jo_texts',
    'VECTOR_INDEX_PATH': BASE_DIR / 'data' / 'jo_vector.index',
    'METADATA_PATH': BASE_DIR / 'data' / 'jo_metadata.pkl',
    
    # Chunking - Optimized for speed
    'CHUNK_SIZE': 400,  # Smaller chunks for faster processing
    'OVERLAP': 30,      # Less overlap for speed
    
    # Retrieval - Fewer results for speed
    'TOP_K_RESULTS': 3,     # Reduced from 5
    'RETRIEVAL_K': 3,       # Reduced from 5
    'CONTEXT_DOCS_COUNT': 2, # Reduced from 3
    
    # Models - Fastest options
    'EMBEDDING_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',  # Faster than multilingual
    'GGUF_MODEL_PATH': BASE_DIR / 'models' / 'SILMA-Kashif-2B-Instruct-v1.0.i1-IQ3_XS.gguf',
    
    # Context and Threading - Optimized for speed
    'CONTEXT_LENGTH': 1024,  # Reduced from 2048
    'N_THREADS': min(os.cpu_count() or 4, 8),  # Use available cores but cap at 8
    
    # Generation - Ultra-fast settings
    'MAX_TOKENS': 80,         # Much shorter responses
    'TEMPERATURE': 0.1,       # Lower for faster, more focused responses
    'TOP_P': 0.7,            # Reduced search space
    'TOP_K': 20,             # Smaller search space
    'REPEAT_PENALTY': 1.05,   # Minimal penalty
    
    # Performance
    'USE_GPU': True,
    'BATCH_SIZE': 1,
    'FAST_MODE': True,  # New flag for speed optimizations
    
    # Legacy (unused)
    'GENERATIVE_MODEL_NAME': 'silma-ai/SILMA-Kashif-2B-Instruct-v1.0',
}

# FIXED Logging Configuration - Shows actual messages
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {module} - {message}',
            'style': '{',
        },
        # Arabic-safe formatter that still shows the actual message
        'arabic_safe': {
            'format': '{levelname} {asctime} {module} - {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',  # Changed from 'safe' to 'simple'
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'rag_app.log',
            'formatter': 'arabic_safe',  # Changed from 'safe' to 'arabic_safe'
            'encoding': 'utf-8',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',  # Changed from WARNING to INFO to see more details
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',  # Changed from WARNING to INFO
            'propagate': False,
        },
        'rag_app': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'llama_cpp': {
            'handlers': ['console'],
            'level': 'ERROR',
            'propagate': False,
        },
        'sentence_transformers': {
            'handlers': ['console'],
            'level': 'ERROR',
            'propagate': False,
        },
        'transformers': {
            'handlers': ['console'],
            'level': 'ERROR',
            'propagate': False,
        },
        'torch': {
            'handlers': ['console'],
            'level': 'ERROR',
            'propagate': False,
        },
    },
}

# FIXED Gemini API Key Configuration
# The issue is here - you're using the actual API key as the first parameter instead of the environment variable name
GEMINI_API_KEY = "AIzaSyA4vBWIrlsbRXXNRXYPepfi3AXK7ada_SQ" # Changed from hardcoded key to env var name