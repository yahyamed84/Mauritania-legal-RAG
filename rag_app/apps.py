from django.apps import AppConfig


class RagAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag_app"
    verbose_name = "RAG Application" # Optional: Human-readable name

    # Optional: If you need to run code when the app is ready (e.g., initialize RAG engine)
    # def ready(self):
    #     # Import the engine here to avoid circular imports
    #     from .rag_engine import rag_engine
    #     if not rag_engine.initialized:
    #          print("Initializing RAG engine from AppConfig...")
    #          rag_engine.initialize()
    #     pass

