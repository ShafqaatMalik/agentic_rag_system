/**
 * Agentic RAG Frontend Application
 * Handles chat, file upload, and API interactions
 */

class AgenticRAG {
    constructor() {
        this.apiBase = '';
        this.isStreaming = false;
        this.conversations = [];
        this.currentConversationId = null;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.loadConversations();
        this.checkHealth();
        this.loadStats();
        this.loadDocuments();

        // Auto-resize textarea
        this.queryInput.addEventListener('input', () => this.autoResize());
    }

    bindElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chat-messages');
        this.chatForm = document.getElementById('chat-form');
        this.queryInput = document.getElementById('query-input');
        this.sendBtn = document.getElementById('send-btn');

        // Health elements
        this.healthIndicator = document.getElementById('health-indicator');
        this.healthDetails = document.getElementById('health-details');

        // Stats elements
        this.docCount = document.getElementById('doc-count');
        this.collectionName = document.getElementById('collection-name');
        this.version = document.getElementById('version');

        // Documents list
        this.documentsList = document.getElementById('documents-list');

        // Conversations list
        this.conversationsList = document.getElementById('conversations-list');

        // Upload elements
        this.uploadZone = document.getElementById('upload-zone');
        this.fileInput = document.getElementById('file-input');
        this.uploadProgress = document.getElementById('upload-progress');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');

        // Actions
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.clearChatBtn = document.getElementById('clear-chat-btn');
    }

    bindEvents() {
        // Form submit
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));

        // Enter to send (Shift+Enter for new line)
        this.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSubmit(e);
            }
        });

        // File upload
        this.uploadZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('dragover');
        });

        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('dragover');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length) this.uploadFile(files[0]);
        });

        // New chat
        this.newChatBtn.addEventListener('click', () => this.createNewConversation());

        // Clear chat
        this.clearChatBtn.addEventListener('click', () => this.deleteCurrentConversation());
    }

    autoResize() {
        this.queryInput.style.height = 'auto';
        this.queryInput.style.height = Math.min(this.queryInput.scrollHeight, 150) + 'px';
    }

    // === Conversation Management ===

    loadConversations() {
        // Load from localStorage
        const saved = localStorage.getItem('rag_conversations');
        if (saved) {
            try {
                this.conversations = JSON.parse(saved);
            } catch (e) {
                console.error('Failed to load conversations:', e);
                this.conversations = [];
            }
        }

        // Create first conversation if none exist
        if (this.conversations.length === 0) {
            this.createNewConversation();
        } else {
            // Load the most recent conversation
            this.currentConversationId = this.conversations[0].id;
            this.renderCurrentConversation();
        }

        this.renderConversations();
    }

    createNewConversation() {
        const conversation = {
            id: Date.now().toString(),
            title: 'New Chat',
            messages: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        this.conversations.unshift(conversation);
        this.currentConversationId = conversation.id;
        this.saveConversations();
        this.renderConversations();
        this.renderCurrentConversation();
    }

    switchConversation(id) {
        if (this.isStreaming) return;

        this.currentConversationId = id;
        this.renderCurrentConversation();
        this.renderConversations(); // Update active state
    }

    deleteCurrentConversation() {
        if (!this.currentConversationId) return;

        if (this.conversations.length === 1) {
            // Don't delete the last conversation, just clear it
            const conv = this.getCurrentConversation();
            conv.messages = [];
            conv.title = 'New Chat';
            conv.updatedAt = new Date().toISOString();
        } else {
            // Remove current conversation
            this.conversations = this.conversations.filter(
                c => c.id !== this.currentConversationId
            );
            // Switch to the first remaining conversation
            this.currentConversationId = this.conversations[0].id;
        }

        this.saveConversations();
        this.renderConversations();
        this.renderCurrentConversation();
    }

    deleteConversation(id) {
        if (this.conversations.length === 1) {
            // Clear instead of delete if it's the last one
            const conv = this.conversations[0];
            conv.messages = [];
            conv.title = 'New Chat';
            conv.updatedAt = new Date().toISOString();
        } else {
            this.conversations = this.conversations.filter(c => c.id !== id);

            // If we deleted the current conversation, switch to another
            if (this.currentConversationId === id) {
                this.currentConversationId = this.conversations[0].id;
                this.renderCurrentConversation();
            }
        }

        this.saveConversations();
        this.renderConversations();
    }

    saveConversations() {
        try {
            localStorage.setItem('rag_conversations', JSON.stringify(this.conversations));
        } catch (e) {
            console.error('Failed to save conversations:', e);
        }
    }

    getCurrentConversation() {
        return this.conversations.find(c => c.id === this.currentConversationId);
    }

    updateConversationTitle(conversation) {
        // Auto-generate title from first user message
        const firstUserMsg = conversation.messages.find(m => m.role === 'user');
        if (firstUserMsg && conversation.title === 'New Chat') {
            conversation.title = firstUserMsg.content.substring(0, 50);
            if (firstUserMsg.content.length > 50) {
                conversation.title += '...';
            }
        }
    }

    renderConversations() {
        if (!this.conversationsList) return;

        if (this.conversations.length === 0) {
            this.conversationsList.innerHTML = `
                <div class="conversations-empty">
                    <p>No conversations yet</p>
                </div>
            `;
            return;
        }

        this.conversationsList.innerHTML = this.conversations.map(conv => {
            const isActive = conv.id === this.currentConversationId;
            const date = new Date(conv.updatedAt);
            const timeStr = this.formatDate(date);

            return `
                <div class="conversation-item ${isActive ? 'active' : ''}" onclick="app.switchConversation('${conv.id}')">
                    <div class="conversation-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                        </svg>
                    </div>
                    <div class="conversation-info">
                        <div class="conversation-title">${this.escapeHtml(conv.title)}</div>
                        <div class="conversation-meta">${timeStr} · ${conv.messages.length} msgs</div>
                    </div>
                    <button class="delete-conversation-btn" onclick="event.stopPropagation(); app.deleteConversation('${conv.id}')" title="Delete conversation">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
            `;
        }).join('');
    }

    renderCurrentConversation() {
        const conversation = this.getCurrentConversation();
        if (!conversation) return;

        // Clear chat
        this.chatMessages.innerHTML = '';

        // Render welcome or messages
        if (conversation.messages.length === 0) {
            this.chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <circle cx="12" cy="12" r="10"/>
                            <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
                            <line x1="9" y1="9" x2="9.01" y2="9"/>
                            <line x1="15" y1="9" x2="15.01" y2="9"/>
                        </svg>
                    </div>
                    <h2>Welcome to Agentic RAG</h2>
                    <p>I'm your AI-powered document assistant with self-correction capabilities.</p>
                    <div class="feature-tags">
                        <span class="tag">Intelligent Routing</span>
                        <span class="tag">Self-Correcting</span>
                        <span class="tag">Hallucination Detection</span>
                    </div>
                    <p class="hint">Upload documents and start asking questions!</p>
                </div>
            `;
        } else {
            // Render all messages
            conversation.messages.forEach(msg => {
                const messageDiv = this.createMessageElement(msg.role, msg.content);
                this.chatMessages.appendChild(messageDiv);

                // Add sources if available
                if (msg.sources && msg.sources.length > 0) {
                    this.addSourcesToMessage(messageDiv, msg.sources);
                }

                // Add timing if available
                if (msg.timing) {
                    this.addTimingToMessage(messageDiv, msg.timing);
                }
            });
        }

        this.scrollToBottom();
    }

    formatDate(date) {
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days < 7) return `${days}d ago`;
        return date.toLocaleDateString();
    }

    // === API Methods ===

    async checkHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();

            this.healthIndicator.classList.remove('healthy', 'unhealthy');
            this.healthIndicator.classList.add(data.status);

            this.healthDetails.innerHTML = `
                <p>Status: ${data.status}</p>
                <p>Vectorstore: ${data.vectorstore_status}</p>
            `;

            this.version.textContent = data.version;
            this.docCount.textContent = data.document_count || 0;

        } catch (error) {
            this.healthIndicator.classList.add('unhealthy');
            this.healthDetails.innerHTML = '<p>Failed to connect</p>';
        }
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.apiBase}/collection/stats`);
            const data = await response.json();

            this.docCount.textContent = data.document_count;
            this.collectionName.textContent = data.name;
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    async loadDocuments() {
        try {
            const response = await fetch(`${this.apiBase}/collection/documents`);
            const data = await response.json();

            if (data.documents && data.documents.length > 0) {
                this.documentsList.innerHTML = data.documents.map(doc => `
                    <div class="document-item">
                        <div class="document-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                            </svg>
                        </div>
                        <div class="document-info">
                            <div class="document-name" title="${doc.name}">${doc.name}</div>
                            <div class="document-meta">${doc.page_count} chunks</div>
                        </div>
                        <button class="delete-doc-btn" onclick="app.deleteDocument('${this.escapeHtml(doc.name)}')" title="Delete document">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"/>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                <line x1="10" y1="11" x2="10" y2="17"/>
                                <line x1="14" y1="11" x2="14" y2="17"/>
                            </svg>
                        </button>
                    </div>
                `).join('');
            } else {
                this.documentsList.innerHTML = `
                    <div class="documents-empty">
                        <p>No documents uploaded yet</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load documents:', error);
            this.documentsList.innerHTML = `
                <div class="documents-empty">
                    <p>Failed to load documents</p>
                </div>
            `;
        }
    }

    async deleteDocument(documentName) {
        if (!confirm(`Are you sure you want to delete "${documentName}"? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/collection/document/${encodeURIComponent(documentName)}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.showToast(`Deleted ${documentName} (${data.chunks_deleted} chunks)`, 'success');
                this.loadDocuments();
                this.loadStats();
                this.checkHealth();
            } else {
                throw new Error(data.message || 'Failed to delete document');
            }
        } catch (error) {
            this.showToast('Failed to delete document: ' + error.message, 'error');
        }
    }

    // === Chat Methods ===

    async handleSubmit(e) {
        e.preventDefault();

        const query = this.queryInput.value.trim();
        if (!query || this.isStreaming) return;

        // Clear welcome message if present
        const welcome = this.chatMessages.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        // Add user message
        this.addMessage('user', query);
        this.queryInput.value = '';
        this.autoResize();

        // Show typing indicator
        const typingId = this.showTypingIndicator();

        this.isStreaming = true;
        this.sendBtn.disabled = true;

        try {
            await this.streamQuery(query, typingId);
        } catch (error) {
            this.removeTypingIndicator(typingId);
            this.addMessage('assistant', 'Sorry, an error occurred. Please try again.');
            this.showToast('Query failed: ' + error.message, 'error');
        } finally {
            this.isStreaming = false;
            this.sendBtn.disabled = false;
        }
    }

    async streamQuery(query, typingId) {
        const response = await fetch(`${this.apiBase}/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, include_sources: true })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let assistantMessageDiv = null;
        let fullContent = '';
        let sources = [];
        let timingData = null;

        // Remove typing indicator and add empty assistant message
        this.removeTypingIndicator(typingId);
        assistantMessageDiv = this.createMessageElement('assistant', '');
        assistantMessageDiv.querySelector('.message-content').innerHTML += '<span class="cursor">|</span>';
        this.chatMessages.appendChild(assistantMessageDiv);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('event:')) {
                    const event = line.replace('event:', '').trim();
                    continue;
                }

                if (line.startsWith('data:')) {
                    try {
                        const data = JSON.parse(line.replace('data:', '').trim());

                        if (data.content) {
                            // Append token incrementally instead of replacing
                            fullContent += data.content;
                            this.updateMessageContent(assistantMessageDiv, fullContent);
                        }

                        if (data.source) {
                            sources.push(data);
                        }

                        if (data.total_ms !== undefined) {
                            timingData = data;
                        }

                        if (data.error) {
                            this.updateMessageContent(assistantMessageDiv, 'Error: ' + data.error);
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete chunks
                    }
                }
            }
        }

        // Add sources if available
        if (sources.length > 0) {
            this.addSourcesToMessage(assistantMessageDiv, sources);
        }

        // Add timing information if available
        if (timingData) {
            this.addTimingToMessage(assistantMessageDiv, timingData);
        }

        // Save assistant message to conversation
        const conversation = this.getCurrentConversation();
        if (conversation) {
            conversation.messages.push({
                role: 'assistant',
                content: fullContent,
                sources: sources.length > 0 ? sources : null,
                timing: timingData || null,
                timestamp: new Date().toISOString()
            });
            conversation.updatedAt = new Date().toISOString();
            this.updateConversationTitle(conversation);
            this.saveConversations();
            this.renderConversations(); // Update conversation list
        }
    }

    addMessage(role, content) {
        // Save to conversation
        const conversation = this.getCurrentConversation();
        if (conversation) {
            conversation.messages.push({
                role: role,
                content: content,
                timestamp: new Date().toISOString()
            });
            conversation.updatedAt = new Date().toISOString();
            this.updateConversationTitle(conversation);
            this.saveConversations();
        }

        // Add to DOM
        const messageDiv = this.createMessageElement(role, content);
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    createMessageElement(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatarIcon = role === 'user'
            ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatarIcon}</div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;

        return messageDiv;
    }

    updateMessageContent(messageDiv, content) {
        const contentDiv = messageDiv.querySelector('.message-content');
        contentDiv.innerHTML = this.formatContent(content);
        this.scrollToBottom();
    }

    addSourcesToMessage(messageDiv, sources) {
        const contentDiv = messageDiv.querySelector('.message-content');

        const sourcesHtml = `
            <div class="sources-section">
                <button class="sources-toggle" onclick="this.classList.toggle('expanded'); this.nextElementSibling.classList.toggle('visible')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                    ${sources.length} source${sources.length > 1 ? 's' : ''} used
                </button>
                <div class="sources-list">
                    ${sources.map(s => `
                        <div class="source-item">
                            <div class="source-name">${this.escapeHtml(s.source || 'Unknown')}</div>
                            <div class="source-content">${this.escapeHtml(s.content || '')}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        contentDiv.insertAdjacentHTML('beforeend', sourcesHtml);
    }

    addTimingToMessage(messageDiv, timingData) {
        const contentDiv = messageDiv.querySelector('.message-content');

        // Format breakdown items
        const breakdownItems = Object.entries(timingData.breakdown || {})
            .map(([nodeName, durationMs]) => {
                const displayName = nodeName.replace(/_/g, ' ');
                return `
                    <div class="timing-item">
                        <span class="timing-label">${displayName}</span>
                        <span class="timing-value">${durationMs}ms</span>
                    </div>
                `;
            })
            .join('');

        const timingHtml = `
            <div class="timing-section">
                <button class="timing-toggle" onclick="this.classList.toggle('expanded'); this.nextElementSibling.classList.toggle('visible')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12 6 12 12 16 14"/>
                    </svg>
                    ${timingData.total_ms}ms total
                </button>
                <div class="timing-breakdown">
                    ${breakdownItems}
                </div>
            </div>
        `;

        contentDiv.insertAdjacentHTML('beforeend', timingHtml);
    }

    showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.id = id;
        typingDiv.className = 'message assistant';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                    <path d="M2 17l10 5 10-5"/>
                    <path d="M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;

        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
        return id;
    }

    removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }

    formatContent(content) {
        // Basic markdown-like formatting
        let formatted = this.escapeHtml(content);

        // Bold
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Code blocks
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

        // Inline code
        formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    // === File Upload Methods ===

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) this.uploadFile(file);
    }

    async uploadFile(file) {
        const allowedTypes = ['.pdf', '.txt', '.md'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedTypes.includes(ext)) {
            this.showToast(`Unsupported file type. Allowed: ${allowedTypes.join(', ')}`, 'error');
            return;
        }

        this.uploadProgress.hidden = false;
        this.progressFill.style.width = '0%';
        this.progressText.textContent = 'Uploading...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 10;
                if (progress <= 90) {
                    this.progressFill.style.width = progress + '%';
                }
            }, 100);

            const response = await fetch(`${this.apiBase}/ingest/file`, {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            this.progressFill.style.width = '100%';

            const data = await response.json();

            if (data.status === 'success') {
                this.progressText.textContent = `Added ${data.chunks_added} chunks`;
                this.showToast(`Successfully uploaded ${file.name}`, 'success');
                this.loadStats();
                this.loadDocuments();
                this.checkHealth();
            } else {
                throw new Error(data.error || 'Upload failed');
            }

        } catch (error) {
            this.progressText.textContent = 'Upload failed';
            this.showToast('Upload failed: ' + error.message, 'error');
        }

        // Hide progress after delay
        setTimeout(() => {
            this.uploadProgress.hidden = true;
            this.fileInput.value = '';
        }, 2000);
    }

    // === Toast Notifications ===

    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span class="toast-message">${message}</span>`;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideUp 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AgenticRAG();
});
