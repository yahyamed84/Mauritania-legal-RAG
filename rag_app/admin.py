from django.contrib import admin
from .models import ChatMessage, DocumentSource, SearchQuery,SystemSettings

# Register your models here to make them accessible in the Django admin interface.
@admin.register(SystemSettings)
class SystemSettingsAdmin(admin.ModelAdmin):
    list_display = ('key', 'value_preview', 'description', 'updated_at', 'created_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('key', 'description', 'value')
    readonly_fields = ('created_at', 'updated_at')
    ordering = ('key',)
    
    def value_preview(self, obj):
        """Show a truncated preview of the value"""
        return obj.value[:50] + '...' if len(obj.value) > 50 else obj.value
    value_preview.short_description = 'Value Preview'

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = (
        "question",
        "answer",
        "confidence",
        "created_at",
        "session_id",
    )
    list_filter = ("created_at", "confidence")
    search_fields = ("question", "answer")
    readonly_fields = ("created_at",)

@admin.register(DocumentSource)
class DocumentSourceAdmin(admin.ModelAdmin):
    list_display = (
        "filename",
        "processed_at",
        "chunk_count",
        "file_size",
        "is_active",
    )
    list_filter = ("is_active", "processed_at")
    search_fields = ("filename", "file_path")
    readonly_fields = ("processed_at",)

@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = (
        "query",
        "results_count",
        "response_time",
        "created_at",
        "user_ip",
    )
    list_filter = ("created_at",)
    search_fields = ("query",)
    readonly_fields = ("created_at",)

