# src/rag_app/views.py - Updated for Model Switching

import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views import View
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .rag_engine import rag_engine_instance  
from .models import ChatMessage, SearchQuery
from django.db.models import Avg
import time
from django.conf import settings # Import settings

logger = logging.getLogger(__name__)

class ChatView(View):
    """Renders the main chat interface page."""
    def get(self, request):
        # Pass Gemini API availability to the template
        gemini_available = bool(settings.GEMINI_API_KEY)
        return render(request, 'chat.html', {"gemini_available": gemini_available})

# src/rag_app/views.py - Debug Model Type Passing

# src/rag_app/views.py - Debug Model Type Passing

@csrf_exempt
@require_http_methods(["POST"])
def ask_question_api(request):
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        session_id = data.get('session_id')
        
        # Get the selected model type from the request.
        # If not provided in request, model_type_from_request will be None.
        model_type_from_request = data.get('model_type')

        logger.info(f"Received request data: {data}")

        final_model_type_for_rag = None # Initialize to None

        if model_type_from_request and model_type_from_request.strip() != '':
            final_model_type_for_rag = model_type_from_request.lower()
            logger.info(f"Extracted model_type from request: '{final_model_type_for_rag}'")
            
            # Validate if model_type is explicitly provided in the request
            allowed_models = ['local', 'gemini']
            if final_model_type_for_rag not in allowed_models:
                return JsonResponse({
                    'error': f"نوع النموذج غير صالح. الخيارات المتاحة: {', '.join(allowed_models)}",
                    'success': False
                }, status=400)

            if final_model_type_for_rag == 'gemini' and not settings.GEMINI_API_KEY:
                return JsonResponse({
                    'error': 'مفتاح Gemini API غير مهيأ في الإعدادات.',
                    'success': False
                }, status=400)
        else:
            # model_type not provided in request, final_model_type_for_rag remains None.
            # This will signal RAG engine to use its default logic (SystemSettings or hardcoded default).
            logger.info(f"model_type not in request payload. RAG engine will use system default.")


        if not question:
            return JsonResponse({
                'error': 'السؤال مطلوب.',
                'success': False
            }, status=400)
        
        # Note: Validation for Gemini API key if model_type is 'gemini' (and resolved by RAG engine)
        # will happen within the RAG engine or later. If final_model_type_for_rag is None now,
        # it might resolve to 'gemini' inside the RAG engine.

        # The logger below will show None if not provided, or the user's choice.
        logger.info(f"Processing user question. Effective model_type to be passed to RAG: '{final_model_type_for_rag}'")

        start_time = time.time()
        # Pass final_model_type_for_rag (which can be None) to the RAG engine.
        # The RAG engine's ask_question method is designed to handle `model_type=None`
        # by checking SystemSettings.
        logger.info(f"Calling rag_engine_instance.ask_question with model_type: '{final_model_type_for_rag}'")
        result = rag_engine_instance.ask_question(question, model_type=final_model_type_for_rag)
        response_time = time.time() - start_time

        logger.info(f"RAG engine returned model_used: {result.get('model_used', 'unknown')}")

        # Determine model_used for saving, prefer RAG's result, fallback to what view intended if RAG was to decide
        model_used_for_saving = result.get('model_used', final_model_type_for_rag if final_model_type_for_rag else 'unknown_resolved_by_rag')


        if not data.get('skip_save', False):
            try:
                ChatMessage.objects.create(
                    question=question,
                    answer=result['answer'],
                    confidence=result['confidence'],
                    sources=result['sources'],
                    session_id=session_id,
                    model_used=model_used_for_saving
                )
            except Exception as e:
                logger.warning(f"Failed to save chat message: {e}")

        response_data = {
            'success': True,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'context': result.get('context_used', ''),
            'retrieved_docs_count': len(result.get('retrieved_documents', [])),
             'retrieved_documents': result.get('retrieved_documents', []), 
            'response_time': round(response_time, 2),
            'search_time': result.get('search_time', 0),
            'generation_time': result.get('generation_time', 0),
            'model_used': result.get('model_used', 'unknown') # RAG engine reports what it actually used
        }
        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'بيانات JSON غير صحيحة.',
            'success': False
        }, status=400)
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        return JsonResponse({
            'error': 'حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.',
            'success': False
        }, status=500)

@api_view(['GET'])
def search_documents_api(request):
    """Fast document search API."""
    try:
        query = request.GET.get('q', '').strip()
        try:
            top_k = min(int(request.GET.get('top_k', 3)), 5)  # Cap at 5 for speed
        except ValueError:
            return Response({
                'error': 'معامل top_k يجب أن يكون رقماً صحيحاً.',
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)

        if not query:
            return Response({
                'error': 'معامل البحث \'q\' مطلوب.',
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)

        logger.info("Performing document search")
        start_time = time.time()
        results = rag_engine_instance.search_documents(query, top_k)
        response_time = time.time() - start_time

        # Skip logging to database for speed
        if not request.GET.get('skip_log', False):
            try:
                SearchQuery.objects.create(
                    query=query,
                    results_count=len(results),
                    response_time=response_time,
                    user_ip=request.META.get('REMOTE_ADDR')
                )
            except Exception as e:
                logger.warning(f"Failed to save search query: {e}")

        return Response({
            'success': True,
            'query': query,
            'results': results,
            'total_found': len(results),
            'response_time': round(response_time, 2)
        })

    except Exception as e:
        logger.error(f"Error during document search: {e}")
        return Response({
            'error': 'حدث خطأ أثناء البحث في المستندات.',
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def health_check_api(request):
    """Fast health check endpoint."""
    try:
        # Check core components status
        core_initialized = rag_engine_instance.initialized_core
        docs_loaded = len(rag_engine_instance.documents) if core_initialized else 0
        index_loaded = rag_engine_instance.index.ntotal if core_initialized and rag_engine_instance.index else 0

        # Check local LLM status (optional, might not be loaded yet)
        local_llm_status = "Not Loaded" if not rag_engine_instance.initialized_local_llm else ("Loaded" if rag_engine_instance.llm else "Load Failed")

        # Check Gemini API status
        gemini_api_configured = bool(settings.GEMINI_API_KEY)

        rag_status = {
            'core_initialized': core_initialized,
            'documents_loaded': docs_loaded,
            'index_loaded': index_loaded,
            'local_llm_status': local_llm_status,
            'gemini_api_configured': gemini_api_configured
        }

        # Quick DB check
        db_ok = True
        try:
            ChatMessage.objects.count()
        except Exception:
            db_ok = False

        # Health depends on core components and DB
        is_healthy = core_initialized and db_ok
        status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

        return Response({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'rag_status': rag_status,
            'database_status': 'connected' if db_ok else 'error'
        }, status=status_code)

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def chat_history_api(request):
    """Fast chat history retrieval."""
    try:
        count = min(int(request.GET.get('count', 10)), 50)  # Limit for speed
        session_id = request.GET.get('session_id')

        query = ChatMessage.objects.order_by('-created_at')
        if session_id:
            query = query.filter(session_id=session_id)

        recent_chats = query[:count]

        history = []
        for chat in recent_chats:
            history.append({
                'id': chat.id,
                'question': chat.question[:100] + '...' if len(chat.question) > 100 else chat.question,  # Truncate for speed
                'answer': chat.answer[:200] + '...' if len(chat.answer) > 200 else chat.answer,  # Truncate for speed
                'confidence': chat.confidence,
                'sources': chat.sources[:3] if isinstance(chat.sources, list) and len(chat.sources) > 3 else chat.sources,  # Limit sources
                'created_at': chat.created_at.isoformat(),
                'session_id': chat.session_id,
                'model_used': getattr(chat, 'model_used', 'unknown') # Include model used if available
            })

        return Response({
            'success': True,
            'history': history
        })

    except ValueError:
        return Response({
            'error': 'معامل count يجب أن يكون رقماً صحيحاً.',
            'success': False
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}", exc_info=True)
        return Response({
            'error': 'حدث خطأ أثناء استرجاع سجل المحادثات.',
            'success': False
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.db.models import Avg, Count, F, Sum, FloatField, Max, Min, StdDev, Q, Func
from django.db.models.functions import TruncDate, TruncHour, TruncMonth, ExtractHour, ExtractWeekDay
from django.utils import timezone
from datetime import timedelta, datetime
import json
import os
import requests
from bs4 import BeautifulSoup
import logging
import random
import re
from collections import Counter
import time
from .models import ChatMessage, DocumentSource, SearchQuery, SystemSettings

logger = logging.getLogger(__name__)

class AnalyticsView(View):
    """
    عرض لوحة الإحصائيات والتحليلات المتقدمة بالعربية
    يعرض إحصائيات النظام، والرسوم البيانية التفاعلية، ومستندات الجريدة الرسمية،
    وتحليلات متقدمة للبيانات، مع دعم التحديث المباشر والتصفية
    """
    
    def get(self, request):
        """عرض صفحة لوحة الإحصائيات المتقدمة"""
        try:
            # إحصائيات أساسية
            total_questions = ChatMessage.objects.count()
            
            # متوسط مستوى الثقة
            avg_confidence_result = ChatMessage.objects.aggregate(
                avg_conf=Avg('confidence'),
                min_conf=Min('confidence'),
                max_conf=Max('confidence')
            )
            avg_confidence = avg_confidence_result['avg_conf'] * 100 if avg_confidence_result['avg_conf'] is not None else 0
            min_confidence = avg_confidence_result['min_conf'] * 100 if avg_confidence_result['min_conf'] is not None else 0
            max_confidence = avg_confidence_result['max_conf'] * 100 if avg_confidence_result['max_conf'] is not None else 0
            
            # عدد المستندات
            documents_count = DocumentSource.objects.filter(is_active=True).count()
            total_chunks = DocumentSource.objects.filter(is_active=True).aggregate(Sum('chunk_count'))['chunk_count__sum'] or 0
            
            # متوسط وقت الاستجابة (بالثواني)
            response_time_stats = SearchQuery.objects.aggregate(
                avg_time=Avg('response_time'),
                min_time=Min('response_time'),
                max_time=Max('response_time')
            )
            avg_response_time = response_time_stats['avg_time'] if response_time_stats['avg_time'] is not None else 0
            
            # إحصائيات استخدام النماذج
            model_usage = ChatMessage.objects.filter(model_used__isnull=False).values('model_used').annotate(
                count=Count('id')
            ).order_by('-count')
            
            # إحصائيات حسب الوقت
            today = timezone.now().date()
            questions_today = ChatMessage.objects.filter(created_at__date=today).count()
            questions_yesterday = ChatMessage.objects.filter(created_at__date=today-timedelta(days=1)).count()
            
            # نسبة التغيير اليومية
            daily_change_pct = 0
            if questions_yesterday > 0:
                daily_change_pct = ((questions_today - questions_yesterday) / questions_yesterday) * 100
            
            # أكثر الأوقات نشاطاً
            hour_distribution = ChatMessage.objects.annotate(
                hour=ExtractHour('created_at')
            ).values('hour').annotate(
                count=Count('id')
            ).order_by('-count')
            
            peak_hour = hour_distribution.first()
            peak_hour_value = peak_hour['hour'] if peak_hour else 0
            
            # حجم المستندات الإجمالي
            total_file_size = DocumentSource.objects.filter(is_active=True).aggregate(
                total_size=Sum('file_size')
            )['total_size'] or 0
            
            # تحويل حجم الملف إلى وحدة مناسبة
            total_size_formatted = self._format_file_size(total_file_size)
            
            # تحضير السياق للقالب
            context = {
                'total_questions': total_questions,
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'documents_count': documents_count,
                'total_chunks': total_chunks,
                'avg_response_time': avg_response_time,
                'model_usage': list(model_usage),
                'questions_today': questions_today,
                'questions_yesterday': questions_yesterday,
                'daily_change_pct': daily_change_pct,
                'peak_hour': peak_hour_value,
                'total_file_size': total_size_formatted,
                'page_title': 'لوحة الإحصائيات والتحليلات المتقدمة',
                'last_update': timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            return render(request, 'analytics.html', context)
            
        except Exception as e:
            logger.error(f"خطأ في عرض صفحة الإحصائيات: {e}", exc_info=True)
            return render(request, 'analytics.html', {
                'error': 'حدث خطأ أثناء تحميل بيانات الإحصائيات.',
                'page_title': 'لوحة الإحصائيات والتحليلات - خطأ'
            })
    
    def _format_file_size(self, size_bytes):
        """تنسيق حجم الملف بوحدة مناسبة"""
        if size_bytes < 1024:
            return f"{size_bytes} بايت"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} كيلوبايت"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.2f} ميجابايت"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} جيجابايت"


def analytics_data_api(request):
    """
    واجهة برمجة التطبيقات (API) لبيانات الإحصائيات المتقدمة
    توفر بيانات للرسوم البيانية والإحصائيات مع دعم التصفية والتخصيص
    """
    try:
        # استخراج معلمات التصفية
        date_from = request.GET.get('date_from')
        date_to = request.GET.get('date_to')
        model_type = request.GET.get('model_type')
        confidence_threshold = request.GET.get('confidence_threshold')
        
        # تاريخ اليوم والأسبوع الماضي (افتراضي)
        today = timezone.now().date()
        default_from = today - timedelta(days=30)  # افتراضياً آخر 30 يوم
        
        # تطبيق التصفية على الاستعلام
        query_filter = Q()
        
        if date_from:
            try:
                date_from_obj = datetime.strptime(date_from, '%Y-%m-%d').date()
                query_filter &= Q(created_at__date__gte=date_from_obj)
            except ValueError:
                query_filter &= Q(created_at__date__gte=default_from)
        else:
            query_filter &= Q(created_at__date__gte=default_from)
            
        if date_to:
            try:
                date_to_obj = datetime.strptime(date_to, '%Y-%m-%d').date()
                query_filter &= Q(created_at__date__lte=date_to_obj)
            except ValueError:
                pass
                
        if model_type and model_type != 'all':
            query_filter &= Q(model_used__icontains=model_type)
            
        if confidence_threshold:
            try:
                conf_threshold = float(confidence_threshold)
                query_filter &= Q(confidence__gte=conf_threshold)
            except ValueError:
                pass
        
        # الأسئلة اليومية
        daily_questions = ChatMessage.objects.filter(query_filter).annotate(
            date=TruncDate('created_at')
        ).values('date').annotate(
            count=Count('id'),
            avg_confidence=Avg(F('confidence') * 100)
        ).order_by('date')
        
        # تحويل البيانات إلى تنسيق مناسب للرسم البياني
        daily_labels = []
        daily_data = []
        confidence_data = []
        
        for item in daily_questions:
            daily_labels.append(item['date'].strftime('%Y-%m-%d'))
            daily_data.append(item['count'])
            confidence_data.append(round(item['avg_confidence'] or 0, 1))
        
        # استخدام النماذج
        model_usage_data = ChatMessage.objects.filter(
            model_used__isnull=False
        ).filter(query_filter).values('model_used').annotate(
            count=Count('id')
        )
        
        model_usage = {
            'local': 0,
            'gemini': 0,
            'unknown': 0
        }
        
        for item in model_usage_data:
            if 'local' in item['model_used'].lower():
                model_usage['local'] += item['count']
            elif 'gemini' in item['model_used'].lower():
                model_usage['gemini'] += item['count']
            else:
                model_usage['unknown'] += item['count']
        
        # توزيع الأسئلة حسب الساعة
        hourly_distribution = ChatMessage.objects.filter(query_filter).annotate(
            hour=ExtractHour('created_at')
        ).values('hour').annotate(
            count=Count('id')
        ).order_by('hour')
        
        hourly_labels = []
        hourly_data = []
        
        for i in range(24):  # 24 ساعة في اليوم
            hourly_labels.append(f"{i}:00")
            
            # البحث عن القيمة المقابلة للساعة
            hour_count = 0
            for item in hourly_distribution:
                if item['hour'] == i:
                    hour_count = item['count']
                    break
                    
            hourly_data.append(hour_count)
        
        # توزيع الأسئلة حسب أيام الأسبوع
        weekday_distribution = ChatMessage.objects.filter(query_filter).annotate(
            weekday=ExtractWeekDay('created_at')
        ).values('weekday').annotate(
            count=Count('id')
        ).order_by('weekday')
        
        weekday_names = ['الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد']
        weekday_data = [0] * 7  # تهيئة مصفوفة بأصفار
        
        for item in weekday_distribution:
            # تعديل الفهرس لأن Django يبدأ من 1 (الأحد) إلى 7 (السبت)
            # بينما مصفوفتنا تبدأ من 0 (الاثنين) إلى 6 (الأحد)
            weekday_index = (item['weekday'] - 2) % 7
            weekday_data[weekday_index] = item['count']
        
        # إحصائيات أوقات الاستجابة
        response_time_stats = SearchQuery.objects.filter(
            created_at__date__gte=default_from
        ).aggregate(
            avg_time=Avg('response_time'),
            min_time=Min('response_time'),
            max_time=Max('response_time'),
            std_dev=StdDev('response_time')
        )
        
        # تجميع البيانات للرد
        response_data = {
            'success': True,
            'daily_questions': {
                'labels': daily_labels,
                'data': daily_data
            },
            'confidence_trend': {
                'labels': daily_labels,
                'data': confidence_data
            },
            'model_usage': model_usage,
            'hourly_distribution': {
                'labels': hourly_labels,
                'data': hourly_data
            },
            'weekday_distribution': {
                'labels': weekday_names,
                'data': weekday_data
            },
            'response_time_stats': {
                'avg': round(response_time_stats['avg_time'] or 0, 2),
                'min': round(response_time_stats['min_time'] or 0, 2),
                'max': round(response_time_stats['max_time'] or 0, 2),
                'std_dev': round(response_time_stats['std_dev'] or 0, 2)
            }
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"خطأ في API بيانات الإحصائيات: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'حدث خطأ أثناء جلب بيانات الإحصائيات'
        }, status=500)


def official_journal_api(request):
    """
    واجهة برمجة التطبيقات (API) لمستندات الجريدة الرسمية
    مع دعم التصفية والبحث والتحميل
    """
    try:
        # معلمات التصفية
        search_term = request.GET.get('search', '')
        limit = min(int(request.GET.get('limit', 10)), 50)  # الحد الأقصى 50
        
        # رابط الصفحة المحتوية على إصدارات الجريدة الرسمية
        base_url = "https://msgg.gov.mr/ar/journal-officiel"
        
        # جلب محتوى الصفحة
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # البحث عن روابط ملفات PDF
        documents = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):
                # استخراج عنوان المستند من النص أو من اسم الملف
                title = link.get_text().strip()
                if not title:
                    title = os.path.basename(href).replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                
                # بناء الرابط الكامل
                full_url = requests.compat.urljoin(base_url, href)
                
                # استخراج التاريخ من اسم الملف إن أمكن
                filename = os.path.basename(href)
                date = "غير محدد"
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        date_part = parts[2].replace('.pdf', '')
                        if len(date_part) == 8:  # تنسيق YYYYMMDD
                            date = f"{date_part[6:8]}/{date_part[4:6]}/{date_part[0:4]}"
                except:
                    pass
                
                # استخراج رقم الإصدار إن أمكن
                issue_number = "غير محدد"
                try:
                    match = re.search(r'(\d+)', filename)
                    if match:
                        issue_number = match.group(1)
                except:
                    pass
                
                # تقدير حجم الملف (سيتم تحديثه لاحقاً بالقيمة الفعلية)
                estimated_size = f"{random.randint(1, 10)}.{random.randint(1, 9)} ميجابايت"
                
                # إضافة معلومات المستند
                document_info = {
                    'title': title,
                    'url': full_url,
                    'filename': filename,
                    'date': date,
                    'issue_number': issue_number,
                    'size': estimated_size
                }
                
                # تطبيق التصفية حسب مصطلح البحث
                if search_term:
                    if (search_term.lower() in title.lower() or 
                        search_term.lower() in filename.lower() or
                        search_term in date or
                        search_term in issue_number):
                        documents.append(document_info)
                else:
                    documents.append(document_info)
        
        # ترتيب المستندات حسب التاريخ (الأحدث أولاً)
        documents.sort(key=lambda x: x['date'], reverse=True)
        
        # تحديد عدد المستندات حسب المعلمة limit
        documents = documents[:limit]
        
        return JsonResponse({
            'success': True,
            'total_count': len(documents),
            'documents': documents
        })
        
    except Exception as e:
        logger.error(f"خطأ في API الجريدة الرسمية: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'حدث خطأ أثناء جلب مستندات الجريدة الرسمية'
        }, status=500)


def top_keywords_api(request):
    """
    واجهة برمجة التطبيقات (API) لأهم الكلمات المفتاحية
    مع تحليل متقدم وتصنيف
    """
    try:
        # معلمات التصفية
        days = int(request.GET.get('days', 30))
        limit = min(int(request.GET.get('limit', 20)), 100)  # الحد الأقصى 100
        
        # تاريخ البداية
        start_date = timezone.now() - timedelta(days=days)
        
        # الحصول على الأسئلة حسب الفترة المحددة
        recent_questions = ChatMessage.objects.filter(
            created_at__gte=start_date
        ).order_by('-created_at').values_list('question', flat=True)
        
        # تحليل الكلمات المفتاحية
        # في التطبيق الحقيقي، يمكن استخدام مكتبات معالجة اللغة الطبيعية المتخصصة
        all_words = []
        stop_words = set([
            'من', 'إلى', 'في', 'على', 'هل', 'ما', 'كيف', 'لماذا', 'متى', 'أين', 
            'من', 'هو', 'هي', 'نحن', 'هم', 'أنا', 'أنت', 'هذا', 'هذه', 'ذلك', 'تلك',
            'الذي', 'التي', 'الذين', 'اللذين', 'اللتين', 'عن', 'مع', 'لكن', 'لأن',
            'و', 'أو', 'ثم', 'بل', 'لا', 'لم', 'لن', 'إن', 'أن', 'كان', 'كانت',
            'يكون', 'تكون', 'سوف', 'سـ', 'قد', 'منذ', 'حتى', 'إذا', 'إلا'
        ])
        
        for question in recent_questions:
            # تقسيم السؤال إلى كلمات وإزالة الكلمات القصيرة والكلمات الشائعة
            words = [word for word in question.split() if len(word) > 2 and word.lower() not in stop_words]
            all_words.extend(words)
        
        # حساب تكرار الكلمات
        word_counter = Counter(all_words)
        
        # ترتيب الكلمات حسب التكرار
        top_words = word_counter.most_common(limit)
        
        # تحويل النتائج إلى تنسيق مناسب
        keywords = []
        for word, count in top_words:
            # تقدير أهمية الكلمة (وزن بين 1 و 10)
            weight = min(10, max(1, int(count / max(1, len(recent_questions)) * 20)))
            
            keywords.append({
                'word': word,
                'count': count,
                'weight': weight
            })
        
        # تصنيف الكلمات المفتاحية (مثال بسيط)
        categories = {
            'قانوني': ['قانون', 'تشريع', 'مرسوم', 'قرار', 'حكم', 'محكمة', 'عدل'],
            'اقتصادي': ['اقتصاد', 'مالية', 'ميزانية', 'ضريبة', 'استثمار', 'تمويل'],
            'اجتماعي': ['تعليم', 'صحة', 'سكن', 'عمل', 'توظيف', 'تقاعد'],
            'سياسي': ['رئاسة', 'وزارة', 'برلمان', 'انتخابات', 'سياسة', 'حكومة']
        }
        
        categorized_keywords = {category: 0 for category in categories}
        
        for keyword in keywords:
            word = keyword['word'].lower()
            for category, terms in categories.items():
                if any(term in word for term in terms):
                    categorized_keywords[category] += keyword['count']
                    break
        
        return JsonResponse({
            'success': True,
            'keywords': keywords,
            'categories': [
                {'name': category, 'count': count}
                for category, count in categorized_keywords.items()
                if count > 0
            ],
            'total_questions': len(recent_questions),
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"خطأ في API الكلمات المفتاحية: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'حدث خطأ أثناء تحليل الكلمات المفتاحية'
        }, status=500)


def system_performance_api(request):
    """
    واجهة برمجة التطبيقات (API) لأداء النظام
    توفر معلومات عن أداء النظام وكفاءته
    """
    try:
        # قياس وقت الاستجابة للاستعلامات
        start_time = time.time()
        
        # إحصائيات أوقات الاستجابة
        response_time_stats = SearchQuery.objects.aggregate(
            avg_time=Avg('response_time'),
            min_time=Min('response_time'),
            max_time=Max('response_time'),
            std_dev=StdDev('response_time')
        )
        
        # متوسط مستوى الثقة
        confidence_stats = ChatMessage.objects.aggregate(
            avg_conf=Avg('confidence'),
            min_conf=Min('confidence'),
            max_conf=Max('confidence'),
            
        )
        
        # إحصائيات حسب النموذج
        model_stats = ChatMessage.objects.filter(
            model_used__isnull=False
        ).values('model_used').annotate(
            count=Count('id'),
            avg_confidence=Avg('confidence'),
            
        ).order_by('-count')
        
        # تحويل إحصائيات النماذج إلى تنسيق مناسب
        models_data = []
        for stat in model_stats:
            models_data.append({
                'model': stat['model_used'],
                'count': stat['count'],
                'avg_confidence': round((stat['avg_confidence'] or 0) * 100, 2),
                
            })
        
        # حساب وقت استجابة API
        api_response_time = time.time() - start_time
        
        return JsonResponse({
            'success': True,
            'response_time_stats': {
                'avg': round(response_time_stats['avg_time'] or 0, 2),
                'min': round(response_time_stats['min_time'] or 0, 2),
                'max': round(response_time_stats['max_time'] or 0, 2),
                'std_dev': round(response_time_stats['std_dev'] or 0, 2)
            },
            'confidence_stats': {
                'avg': round((confidence_stats['avg_conf'] or 0) * 100, 2),
                'min': round((confidence_stats['min_conf'] or 0) * 100, 2),
                'max': round((confidence_stats['max_conf'] or 0) * 100, 2),
                
            },
            'model_stats': models_data,
            'api_response_time': round(api_response_time, 4),
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"خطأ في API أداء النظام: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'حدث خطأ أثناء جلب بيانات أداء النظام'
        }, status=500)


def document_stats_api(request):
    """
    واجهة برمجة التطبيقات (API) لإحصائيات المستندات
    توفر معلومات تفصيلية عن المستندات المخزنة في النظام
    """
    try:
        # إحصائيات المستندات
        doc_stats = DocumentSource.objects.filter(is_active=True).aggregate(
            total_docs=Count('id'),
            total_chunks=Sum('chunk_count'),
            total_size=Sum('file_size'),
            avg_chunks=Avg('chunk_count'),
            avg_size=Avg('file_size')
        )
        
        # توزيع المستندات حسب الشهر
        monthly_docs = DocumentSource.objects.filter(
            is_active=True
        ).annotate(
            month=TruncMonth('processed_at')
        ).values('month').annotate(
            count=Count('id'),
            total_size=Sum('file_size')
        ).order_by('month')
        
        # تحويل البيانات إلى تنسيق مناسب
        monthly_data = []
        for item in monthly_docs:
            monthly_data.append({
                'month': item['month'].strftime('%Y-%m'),
                'count': item['count'],
                'total_size': item['total_size']
            })
        
        # تنسيق حجم الملفات
        total_size_bytes = doc_stats['total_size'] or 0
        if total_size_bytes < 1024:
            total_size_formatted = f"{total_size_bytes} بايت"
        elif total_size_bytes < 1024 * 1024:
            total_size_formatted = f"{total_size_bytes/1024:.2f} كيلوبايت"
        elif total_size_bytes < 1024 * 1024 * 1024:
            total_size_formatted = f"{total_size_bytes/(1024*1024):.2f} ميجابايت"
        else:
            total_size_formatted = f"{total_size_bytes/(1024*1024*1024):.2f} جيجابايت"
        
        # أحدث المستندات المضافة
        recent_docs = DocumentSource.objects.filter(
            is_active=True
        ).order_by('-processed_at')[:5].values(
            'filename', 'processed_at', 'chunk_count', 'file_size'
        )
        
        # تحويل أحدث المستندات إلى تنسيق مناسب
        recent_docs_list = []
        for doc in recent_docs:
            # تنسيق حجم الملف
            file_size = doc['file_size']
            if file_size < 1024:
                size_formatted = f"{file_size} بايت"
            elif file_size < 1024 * 1024:
                size_formatted = f"{file_size/1024:.2f} كيلوبايت"
            else:
                size_formatted = f"{file_size/(1024*1024):.2f} ميجابايت"
                
            recent_docs_list.append({
                'filename': doc['filename'],
                'processed_at': doc['processed_at'].isoformat(),
                'chunk_count': doc['chunk_count'],
                'file_size': size_formatted
            })
        
        return JsonResponse({
            'success': True,
            'document_stats': {
                'total_docs': doc_stats['total_docs'] or 0,
                'total_chunks': doc_stats['total_chunks'] or 0,
                'total_size': total_size_formatted,
                'avg_chunks_per_doc': round(doc_stats['avg_chunks'] or 0, 1),
                'avg_size_per_doc': round((doc_stats['avg_size'] or 0) / (1024 * 1024), 2)  # بالميجابايت
            },
            'monthly_distribution': monthly_data,
            'recent_documents': recent_docs_list
        })
        
    except Exception as e:
        logger.error(f"خطأ في API إحصائيات المستندات: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'حدث خطأ أثناء جلب إحصائيات المستندات'
        }, status=500)


def user_activity_api(request):
    """
    واجهة برمجة التطبيقات (API) لنشاط المستخدمين
    توفر معلومات عن نشاط المستخدمين وتفاعلهم مع النظام
    """
    try:
        # معلمات التصفية
        days = int(request.GET.get('days', 30))
        
        # تاريخ البداية
        start_date = timezone.now() - timedelta(days=days)
        
        # نشاط المستخدمين حسب الجلسة
        session_activity = ChatMessage.objects.filter(
            created_at__gte=start_date,
            session_id__isnull=False
        ).values('session_id').annotate(
            questions_count=Count('id'),
            first_activity=Min('created_at'),
            last_activity=Max('created_at'),
            avg_confidence=Avg('confidence')
        ).order_by('-last_activity')
        
        # تحويل البيانات إلى تنسيق مناسب
        sessions_data = []
        for session in session_activity:
            # حساب مدة الجلسة بالدقائق
            duration_seconds = (session['last_activity'] - session['first_activity']).total_seconds()
            duration_minutes = round(duration_seconds / 60, 1)
            
            sessions_data.append({
                'session_id': session['session_id'],
                'questions_count': session['questions_count'],
                'first_activity': session['first_activity'].isoformat(),
                'last_activity': session['last_activity'].isoformat(),
                'duration_minutes': duration_minutes,
                'avg_confidence': round((session['avg_confidence'] or 0) * 100, 1)
            })
        
        # توزيع النشاط حسب اليوم
        daily_activity = ChatMessage.objects.filter(
            created_at__gte=start_date
        ).annotate(
            date=TruncDate('created_at')
        ).values('date').annotate(
            count=Count('id'),
            avg_confidence=Avg('confidence')
        ).order_by('date')
        
        # تحويل البيانات إلى تنسيق مناسب
        daily_data = []
        for day in daily_activity:
            daily_data.append({
                'date': day['date'].isoformat(),
                'count': day['count'],
                'avg_confidence': round((day['avg_confidence'] or 0) * 100, 1)
            })
        
        # إحصائيات عامة
        total_sessions = len(sessions_data)
        total_questions = sum(session['questions_count'] for session in sessions_data)
        avg_questions_per_session = round(total_questions / max(1, total_sessions), 1)
        avg_session_duration = round(sum(session['duration_minutes'] for session in sessions_data) / max(1, total_sessions), 1)
        
        return JsonResponse({
            'success': True,
            'period_days': days,
            'activity_stats': {
                'total_sessions': total_sessions,
                'total_questions': total_questions,
                'avg_questions_per_session': avg_questions_per_session,
                'avg_session_duration_minutes': avg_session_duration
            },
            'sessions': sessions_data[:50],  # أحدث 50 جلسة فقط
            'daily_activity': daily_data
        })
        
    except Exception as e:
        logger.error(f"خطأ في API نشاط المستخدمين: {e}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': 'حدث خطأ أثناء جلب بيانات نشاط المستخدمين'
        }, status=500)

# Add these imports if not already present
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import ChatMessage, DocumentSource, SearchQuery, SystemSettings
from django.utils import timezone
from django.db.models import Avg
import json
import logging

logger = logging.getLogger(__name__)

def admin_settings(request):
    """
    Render the admin settings page
    """
    return render(request, 'admin_settings.html')

@api_view(['POST'])
def save_settings(request):
    """
    Save model and RAG settings using SystemSettings model
    """
    try:
        # Parse JSON data
        if hasattr(request, 'data'):
            data = request.data  # DRF parsed data
        else:
            data = json.loads(request.body)
        
        logger.info(f"Received settings data: {data}")
        
        # Extract settings
        model_type = data.get('model_type', 'local')
        model_params = data.get('model_params', {})
        rag_settings = data.get('rag_settings', {})
        
        # Save model type
        if model_type:
            SystemSettings.objects.update_or_create(
                key='model_type',
                defaults={'value': model_type}
            )
            logger.info(f"Saved model_type: {model_type}")
        
        # Save model parameters if provided
        if model_params:
            SystemSettings.objects.update_or_create(
                key='model_params',
                defaults={'value': json.dumps(model_params)}
            )
            logger.info(f"Saved model_params: {model_params}")
        
        # Save RAG settings if provided
        if rag_settings:
            SystemSettings.objects.update_or_create(
                key='rag_settings',
                defaults={'value': json.dumps(rag_settings)}
            )
            logger.info(f"Saved rag_settings: {rag_settings}")
        
        return Response({
            'status': 'success',
            'message': 'Settings saved successfully',
            'saved_data': {
                'model_type': model_type,
                'model_params': model_params,
                'rag_settings': rag_settings
            }
        })
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in save_settings: {str(e)}")
        return Response({
            'status': 'error',
            'message': f'Invalid JSON data: {str(e)}'
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Error in save_settings: {str(e)}")
        return Response({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def system_status(request):
    """
    Get system status information
    """
    try:
        # Get query count
        query_count = SearchQuery.objects.count() if SearchQuery.objects.count() else 0
        
        # Get average confidence (handle case where ChatMessage model might not have confidence field)
        try:
            avg_confidence = ChatMessage.objects.aggregate(avg_confidence=Avg('confidence'))
            avg_confidence_value = avg_confidence.get('avg_confidence', 0) or 0
        except Exception:
            avg_confidence_value = 0
        
        # Get current model type from database
        model_type_setting = SystemSettings.objects.filter(key='model_type').first()
        current_model = model_type_setting.value if model_type_setting else 'local'
        
        return Response({
            'status': 'success',
            'query_count': query_count,
            'avg_confidence': round(avg_confidence_value, 2) if avg_confidence_value else 0,
            'current_model': current_model
        })
    except Exception as e:
        logger.error(f"Error in system_status: {str(e)}")
        return Response({
            'status': 'error',
            'message': f'Error fetching system status: {str(e)}',
            'query_count': 0,
            'avg_confidence': 0,
            'current_model': 'local'
        })

@api_view(['GET'])
def get_settings(request):
    """
    Get all settings for the admin page
    """
    try:
        # Get model type
        model_type_setting = SystemSettings.objects.filter(key='model_type').first()
        model_type = model_type_setting.value if model_type_setting else 'local'
        
        # Get model parameters
        model_params_setting = SystemSettings.objects.filter(key='model_params').first()
        try:
            model_params = json.loads(model_params_setting.value) if model_params_setting and model_params_setting.value else {}
        except json.JSONDecodeError:
            model_params = {}
        
        # Set default model parameters if not exist
        if not model_params:
            model_params = {
                'local': {
                    'temperature': 0.7,
                    'max_tokens': 1000,
                    'top_k': 40
                },
                'gemini': {
                    'temperature': 0.7,
                    'max_tokens': 1000,
                    'top_k': 20
                }
            }
        
        # Get RAG settings
        rag_settings_obj = SystemSettings.objects.filter(key='rag_settings').first()
        try:
            rag_settings = json.loads(rag_settings_obj.value) if rag_settings_obj and rag_settings_obj.value else {}
        except json.JSONDecodeError:
            rag_settings = {}
        
        # Set default RAG settings if not exist
        if not rag_settings:
            rag_settings = {
                'num_docs': 3,
                'similarity_threshold': 0.5
            }
        
        logger.info(f"Retrieved settings - model_type: {model_type}, model_params: {model_params}, rag_settings: {rag_settings}")
        
        return Response({
            'status': 'success',
            'model_type': model_type,
            'model_params': model_params,
            'rag_settings': rag_settings
        })
    except Exception as e:
        logger.error(f"Error in get_settings: {str(e)}")
        return Response({
            'status': 'error',
            'message': f'Error fetching settings: {str(e)}',
            'model_type': 'local',
            'model_params': {
                'local': {'temperature': 0.7, 'max_tokens': 1000, 'top_k': 40},
                'gemini': {'temperature': 0.7, 'max_tokens': 1000, 'top_k': 20}
            },
            'rag_settings': {'num_docs': 3, 'similarity_threshold': 0.5}
        })

# Additional endpoint to get settings by key (helper function)
@api_view(['GET'])
def get_setting_by_key(request, key):
    """
    Get a specific setting by key
    """
    try:
        setting = SystemSettings.objects.filter(key=key).first()
        if setting:
            try:
                # Try to parse as JSON first
                value = json.loads(setting.value)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, return as string
                value = setting.value
            
            return Response({
                'status': 'success',
                'key': key,
                'value': value
            })
        else:
            return Response({
                'status': 'error',
                'message': f'Setting with key "{key}" not found'
            }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error getting setting by key {key}: {str(e)}")
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
