from django.urls import path
from rag_app import views
from django.contrib import admin
app_name = "rag_app"

urlpatterns = [
    # Main chat interface page (rendered by a class-based view)
    path("", views.ChatView.as_view(), name="chat"),
path("admin/", admin.site.urls, name="admin"),
    # API endpoints (functional views)
    path("api/ask/", views.ask_question_api, name="ask_question_api"),
    path("api/search/", views.search_documents_api, name="search_documents_api"),
    path("api/health/", views.health_check_api, name="health_check_api"),
    path("api/history/", views.chat_history_api, name="chat_history_api"),
    
      # New URL patterns for admin setting
    path('admin-settings/', views.admin_settings, name='admin_settings'),
    path('api/save-settings/', views.save_settings, name='save_settings'),
    path('api/system-status/', views.system_status, name='system_status'),
    path('api/get-settings/', views.get_settings, name='get_settings'),
    path('api/get-setting/<str:key>/', views.get_setting_by_key, name='get_setting_by_key'),

    # Analytics page (rendered by a class-based view)
      # مسار صفحة الإحصائيات المتقدمة
    path("analytics/", views.AnalyticsView.as_view(), name="analytics"),
    
    # واجهات برمجة التطبيقات (APIs) للإحصائيات المتقدمة
    path("api/analytics-data/", views.analytics_data_api, name="analytics_data_api"),
    path("api/official-journal/", views.official_journal_api, name="official_journal_api"),
    path("api/top-keywords/", views.top_keywords_api, name="top_keywords_api"),
    path("api/system-performance/", views.system_performance_api, name="system_performance_api"),
    path("api/document-stats/", views.document_stats_api, name="document_stats_api"),
    path("api/user-activity/", views.user_activity_api, name="user_activity_api"),

]

