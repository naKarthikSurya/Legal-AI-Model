// model.js - JavaScript for the ChatGPT-style model interface
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const chatMessages = document.getElementById('chat-messages');
    const loadingIndicator = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');
    const clearHistoryBtn = document.getElementById('clear-history');
    
    // Chat history storage - limited to 4 recent conversations
    const MAX_HISTORY_ITEMS = 4;
    let chatHistoryItems = loadChatHistory();
    
    // Function to load chat history from localStorage
    function loadChatHistory() {
        const savedHistory = localStorage.getItem('legalChatHistory');
        if (savedHistory) {
            try {
                return JSON.parse(savedHistory);
            } catch (e) {
                console.error("Error parsing chat history:", e);
                return [];
            }
        }
        return [];
    }
    
    // Function to save chat history to localStorage
    function saveChatHistory() {
        // Limit history to MAX_HISTORY_ITEMS
        chatHistoryItems = chatHistoryItems.slice(-MAX_HISTORY_ITEMS);
        localStorage.setItem('legalChatHistory', JSON.stringify(chatHistoryItems));
    }
    
    // Function to display chat history
    function displayChatHistory() {
        // Clear existing messages except for system welcome message
        const systemMessage = chatMessages.querySelector('.system-message');
        chatMessages.innerHTML = '';
        
        // Re-add system message
        if (systemMessage) {
            chatMessages.appendChild(systemMessage);
        }
        
        // Add saved chat history
        chatHistoryItems.forEach(item => {
            addMessageToChat('user', item.query);
            addMessageToChat('assistant', item.response);
        });
        
        // Scroll to bottom
        scrollToBottom();
    }
    
    // Function to add a message to the chat
    function addMessageToChat(sender, content) {
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${sender}-message`;
        
        const iconEl = document.createElement('div');
        iconEl.className = 'message-icon';
        
        if (sender === 'user') {
            iconEl.innerHTML = '<i class="fas fa-user"></i>';
        } else if (sender === 'assistant') {
            iconEl.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        contentEl.innerHTML = formatMessageContent(content);
        
        messageEl.appendChild(iconEl);
        messageEl.appendChild(contentEl);
        
        chatMessages.appendChild(messageEl);
        scrollToBottom();
    }
    
    // Format message content with paragraphs and handling for sources
    function formatMessageContent(content) {
        // If content is a string, format it
        if (typeof content === 'string') {
            return content
                .split('\n\n')
                .map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
                .join('');
        }
        
        // If content is an object with response and sources
        if (content.response) {
            let formattedContent = content.response
                .split('\n\n')
                .map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
                .join('');
            
            // Add sources if available
            if (content.sources && content.sources.length > 0) {
                formattedContent += '<div class="sources-section"><h4>Sources:</h4><ul>';
                
                content.sources.forEach((source, index) => {
                    const fileName = source.metadata && source.metadata.file_name 
                        ? source.metadata.file_name 
                        : `Source ${index + 1}`;
                    
                    formattedContent += `<li><strong>${fileName}</strong>: ${truncateText(source.content, 150)}</li>`;
                });
                
                formattedContent += '</ul></div>';
            }
            
            return formattedContent;
        }
        
        // Fallback
        return `<p>${content}</p>`;
    }
    
    // Function to scroll the chat to the bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Show chat history on page load
    displayChatHistory();
    
    // Function to submit query to API
    async function submitQuery(query) {
        // Add user message to chat
        addMessageToChat('user', query);
        
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        
        // Disable input during processing
        queryInput.disabled = true;
        submitBtn.disabled = true;
        
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add to chat history
            const historyItem = {
                query: query,
                response: data.response,
                timestamp: Date.now(),
                sources: data.sources || []
            };
            
            chatHistoryItems.push(historyItem);
            saveChatHistory();
            
            // Add assistant response to chat
            addMessageToChat('assistant', {
                response: data.response,
                sources: data.sources
            });
            
        } catch (error) {
            console.error('Error:', error);
            errorMessage.classList.remove('hidden');
        } finally {
            loadingIndicator.classList.add('hidden');
            queryInput.disabled = false;
            submitBtn.disabled = false;
            queryInput.focus();
        }
    }
    
    // Truncate text for source display
    function truncateText(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
    
    // Form submission event
    queryForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const query = queryInput.value.trim();
        
        if (query) {
            submitQuery(query);
            queryInput.value = ''; // Clear input after submission
        }
    });
    
    // Clear chat history
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to clear your conversation history?')) {
                chatHistoryItems = [];
                saveChatHistory();
                displayChatHistory();
            }
        });
    }
    
    // Auto-resize input field as user types
    if (queryInput) {
        queryInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
});