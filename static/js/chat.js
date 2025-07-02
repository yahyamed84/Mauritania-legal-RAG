let isLoading = false;
let chatContainer;
let questionInput;
let sendBtn;
let welcomeMessage;

// Function to get CSRF token from cookies
function getCSRFToken() {
    return document.cookie.match(/csrftoken=([^;]+)/)?.[1] ||
           document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
}

// Function to escape HTML special characters
function escapeHtml(text) {
    if (text === null || text === undefined) {
        return '';
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Function to format the bot's answer (e.g., handle newlines)
function formatAnswer(answer) {
    // Basic formatting for Arabic text
    return escapeHtml(answer)
        .replace(/\n/g, '<br>') // Replace newline characters with <br>
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Basic bold formatting
}

// Function to scroll the chat container to the bottom
function scrollToBottom() {
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

// Function to add a user message to the chat
function addUserMessage(question) {
    if (!chatContainer) return;

    if (welcomeMessage && welcomeMessage.parentNode === chatContainer) {
        chatContainer.removeChild(welcomeMessage);
        welcomeMessage = null;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `
        <div class="message-bubble">
            <i class="fas fa-user me-2"></i>
            ${escapeHtml(question)}
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// MODIFIED Function to add a bot message to the chat
// Expects 'retrievedDocuments' which is an array of objects
function addBotMessage(answer, confidence, retrievedDocuments) {
    if (!chatContainer) return;

    const confidenceClass = confidence > 0.7 ? 'confidence-high' :
                           confidence > 0.4 ? 'confidence-medium' : 'confidence-low';
    const confidenceText = confidence > 0.7 ? 'ثقة عالية' :
                          confidence > 0.4 ? 'ثقة متوسطة' : 'ثقة منخفضة';

    let sourcesHtml = '';
    // Check if retrievedDocuments is provided and is an array with items
    if (retrievedDocuments && Array.isArray(retrievedDocuments) && retrievedDocuments.length > 0) {
        sourcesHtml = `
            <div class="sources">
                <strong><i class="fas fa-book me-1"></i>المصادر المسترجعة:</strong>
                ${retrievedDocuments.map(doc => {
                    // Ensure doc and its properties exist before trying to access them
                    const sourceName = doc && doc.source ? escapeHtml(doc.source) : 'مصدر غير معروف';
                    const rank = doc && doc.rank ? doc.rank : 'N/A';
                    const similarity = doc && typeof doc.similarity_score === 'number' ? doc.similarity_score.toFixed(2) : 'N/A';
                    return `
                        <div class="source-item">
                            <i class="fas fa-file-alt text-secondary me-1"></i>
                            <span>${sourceName}</span>
                            <small class="text-muted ms-2">(ترتيب: ${rank}, تشابه: ${similarity})</small>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.innerHTML = `
        <div class="message-bubble">
            <div class="d-flex align-items-center mb-2">
                <i class="fas fa-robot text-primary me-2"></i>
                <span class="fw-bold">المساعد الذكي</span>
                <span class="confidence-badge ${confidenceClass} ms-auto">
                    ${confidenceText} (${Math.round(confidence * 100)}%)
                </span>
            </div>
            <div>${formatAnswer(answer)}</div>
            ${sourcesHtml}
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Function to show the typing indicator (no changes)
function showTypingIndicator() {
    if (!chatContainer) return;
    removeTypingIndicator();
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <i class="fas fa-robot text-primary"></i>
        <span>المساعد يكتب</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatContainer.appendChild(typingDiv);
    scrollToBottom();
}

// Function to remove the typing indicator (no changes)
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator && typingIndicator.parentNode === chatContainer) {
        chatContainer.removeChild(typingIndicator);
    }
}

// Function to show an error message in the chat (no changes)
function showError(message) {
    if (!chatContainer) return;
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${escapeHtml(message)}
    `;
    chatContainer.appendChild(errorDiv);
    scrollToBottom();
    setTimeout(() => {
        if (errorDiv.parentNode === chatContainer) {
            chatContainer.removeChild(errorDiv);
        }
    }, 5000);
}

// Function to set the loading state (no changes)
function setLoading(loading) {
    isLoading = loading;
    if (sendBtn) sendBtn.disabled = loading;
    if (questionInput) questionInput.disabled = loading;
    if (sendBtn) {
        sendBtn.innerHTML = loading ? '<i class="fas fa-spinner fa-spin"></i>' : '<i class="fas fa-paper-plane"></i>';
    }
}

// Main function to handle asking a question
async function askQuestion() {
    if (isLoading || !questionInput) return;
    const question = questionInput.value.trim();

    if (!question) {
        showError('يرجى كتابة سؤال أولاً');
        return;
    }
    if (question.length > 500) {
        showError('السؤال طويل جداً. الحد الأقصى 500 حرف.');
        return;
    }

    questionInput.value = '';
    addUserMessage(question);
    showTypingIndicator();
    setLoading(true);

    try {
        const response = await fetch('/api/ask/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({ question: question })
        });

        removeTypingIndicator();

        if (!response.ok) {
             let errorMsg = `خطأ ${response.status}: ${response.statusText}`;
             try {
                 const errorData = await response.json();
                 errorMsg = errorData.error || errorMsg;
             } catch (e) { /* Ignore if response is not JSON */ }
             showError(errorMsg);
        } else {
            const data = await response.json();
            console.log("Received data from server:", data); // Log received data for debugging
            if (data.success) {
                // MODIFIED: Pass data.retrieved_documents to addBotMessage
                addBotMessage(data.answer, data.confidence, data.retrieved_documents);
            } else {
                showError(data.error || 'حدث خطأ غير متوقع في الاستجابة');
            }
        }
    } catch (error) {
        console.error('Fetch Error:', error);
        removeTypingIndicator();
        showError('خطأ في الاتصال بالخادم. يرجى المحاولة مرة أخرى.');
    } finally {
        setLoading(false);
        if (questionInput) {
             questionInput.focus();
        }
    }
}

// Function to handle suggestion clicks (no changes)
function askSuggestion(question) {
    if (questionInput) {
        questionInput.value = question;
        askQuestion();
    }
}

// Initialize script when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    chatContainer = document.getElementById('chatContainer');
    questionInput = document.getElementById('questionInput');
    sendBtn = document.getElementById('sendBtn');
    welcomeMessage = document.getElementById('welcomeMessage');

    if (questionInput) {
        questionInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
        questionInput.focus();
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', askQuestion);
    }

    // This CSRF input creation is a fallback, generally not needed if X-CSRFToken header is correctly set.
    if (!document.querySelector('[name=csrfmiddlewaretoken]')) {
        const csrfInput = document.createElement('input');
        csrfInput.type = 'hidden';
        csrfInput.name = 'csrfmiddlewaretoken';
        csrfInput.value = getCSRFToken();
        if (document.body) { // Ensure body exists before appending
             document.body.appendChild(csrfInput);
        } else { // Fallback if body isn't ready (shouldn't happen with DOMContentLoaded)
            document.documentElement.appendChild(csrfInput);
        }
    }
});