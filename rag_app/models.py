# src/rag_app/models.py - Added model_used field

from django.db import models
from django.utils import timezone

class ChatMessage(models.Model):
    """Model to store chat interactions"""
    question = models.TextField(verbose_name="السؤال")
    answer = models.TextField(verbose_name="الجواب")
    confidence = models.FloatField(default=0.0, verbose_name="مستوى الثقة" ,null=True, blank=True)
    # Store sources as JSON. Ensure the data passed is serializable (list/dict).
    sources = models.JSONField(default=list, verbose_name="المصادر")
    created_at = models.DateTimeField(default=timezone.now, verbose_name="وقت الإنشاء")
    session_id = models.CharField(max_length=100, blank=True, null=True, verbose_name="معرف الجلسة")
    # New field to track which model generated the answer
    model_used = models.CharField(max_length=50, blank=True, null=True, verbose_name="النموذج المستخدم") # e.g., 'local_gguf', 'gemini_api', 'local_error'

    class Meta:
        verbose_name = "رسالة المحادثة"
        verbose_name_plural = "رسائل المحادثة"
        ordering = [
            "-created_at"
        ]  # Order by most recent first

    def __str__(self):
        return f'{self.question[:50]}... ({self.model_used or "unknown"}) - {self.created_at.strftime("%Y-%m-%d %H:%M")}'

class DocumentSource(models.Model):
    """Model to track document sources (Optional, if you manage sources in DB)"""
    filename = models.CharField(max_length=255, unique=True, verbose_name="اسم الملف")
    file_path = models.CharField(max_length=500, verbose_name="مسار الملف")
    processed_at = models.DateTimeField(default=timezone.now, verbose_name="وقت المعالجة")
    chunk_count = models.IntegerField(default=0, verbose_name="عدد القطع")
    file_size = models.BigIntegerField(default=0, verbose_name="حجم الملف")
    is_active = models.BooleanField(default=True, verbose_name="نشط")

    class Meta:
        verbose_name = "مصدر المستند"
        verbose_name_plural = "مصادر المستندات"
        ordering = ["-processed_at"]

    def __str__(self):
        return self.filename

class SearchQuery(models.Model):
    """Model to track search queries for analytics"""
    query = models.TextField(verbose_name="استعلام البحث")
    results_count = models.IntegerField(default=0, verbose_name="عدد النتائج")
    response_time = models.FloatField(default=0.0, verbose_name="وقت الاستجابة")
    created_at = models.DateTimeField(default=timezone.now, verbose_name="وقت البحث")
    user_ip = models.GenericIPAddressField(blank=True, null=True, verbose_name="عنوان IP")

    class Meta:
        verbose_name = "استعلام البحث"
        verbose_name_plural = "استعلامات البحث"
        ordering = ["-created_at"]

    def __str__(self):
        return f'{self.query[:30]}... - {self.created_at.strftime("%Y-%m-%d")}'

# Add this to your existing models.py file

# Add this to your models.py file

from django.db import models
from django.utils import timezone

class SystemSettings(models.Model):
    """
    Model to store system configuration settings as key-value pairs
    """
    key = models.CharField(max_length=100, unique=True, db_index=True)
    value = models.TextField()  # Store as JSON string for complex data
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'system_settings'
        verbose_name = 'System Setting'
        verbose_name_plural = 'System Settings'
        ordering = ['key']
    
    def __str__(self):
        return f"{self.key}: {self.value[:50]}..."
    
    @classmethod
    def get_setting(cls, key, default=None):
        """
        Helper method to get a setting value by key
        """
        try:
            setting = cls.objects.get(key=key)
            return setting.value
        except cls.DoesNotExist:
            return default
    
    @classmethod
    def set_setting(cls, key, value, description=None):
        """
        Helper method to set a setting value by key
        """
        setting, created = cls.objects.update_or_create(
            key=key,
            defaults={
                'value': value,
                'description': description
            }
        )
        return setting