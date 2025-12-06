document.addEventListener('DOMContentLoaded', () => {
    // Search tab elements
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const loadingDiv = document.getElementById('loading');
    const answerDiv = document.getElementById('answer');
    const matchesDiv = document.getElementById('matches');

    // Single ingest tab elements
    const ingestInput = document.getElementById('ingestInput');
    const ingestTopic = document.getElementById('ingestTopic');
    const ingestBtn = document.getElementById('ingestBtn');
    const ingestStatus = document.getElementById('ingestStatus');

    // Bulk upload tab elements
    const bulkInput = document.getElementById('bulkInput');
    const bulkTopic = document.getElementById('bulkTopic');
    const bulkBtn = document.getElementById('bulkBtn');
    const bulkStatus = document.getElementById('bulkStatus');
    const urlCount = document.getElementById('urlCount');

    // Library tab elements
    const topicFilter = document.getElementById('topicFilter');
    const authorFilterLibrary = document.getElementById('authorFilterLibrary');
    const refreshLibrary = document.getElementById('refreshLibrary');
    const libraryLoading = document.getElementById('libraryLoading');
    const libraryStats = document.getElementById('libraryStats');
    const libraryVideos = document.getElementById('libraryVideos');
    const categoryNav = document.getElementById('categoryNav');
    const librarySortBy = document.getElementById('librarySortBy');

    // Category management elements
    const manageCategoriesBtn = document.getElementById('manageCategoriesBtn');
    const categoryModal = document.getElementById('categoryModal');
    const closeCategoryModal = document.getElementById('closeCategoryModal');
    const newCategoryInput = document.getElementById('newCategoryInput');
    const addCategoryBtn = document.getElementById('addCategoryBtn');
    const categoryList = document.getElementById('categoryList');

    // Video category edit elements
    const videoCategoryModal = document.getElementById('videoCategoryModal');
    const closeVideoCategoryModal = document.getElementById('closeVideoCategoryModal');
    const editVideoTitle = document.getElementById('editVideoTitle');
    const videoCategoryCheckboxes = document.getElementById('videoCategoryCheckboxes');
    const saveVideoCategoriesBtn = document.getElementById('saveVideoCategoriesBtn');

    // Video edit modal elements
    const videoEditModal = document.getElementById('videoEditModal');
    const closeVideoEditModal = document.getElementById('closeVideoEditModal');
    const editTitleInput = document.getElementById('editTitleInput');
    const editKeyTakeawayInput = document.getElementById('editKeyTakeawayInput');
    const editSummaryInput = document.getElementById('editSummaryInput');
    const saveVideoEditBtn = document.getElementById('saveVideoEditBtn');

    // Tab elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Video URL regex - supports TikTok, YouTube, Instagram, Twitter/X, Facebook, Vimeo, Reddit
    const VIDEO_URL_REGEX = /https?:\/\/(?:www\.)?(?:tiktok\.com|youtube\.com|youtu\.be|instagram\.com|twitter\.com|x\.com|facebook\.com|fb\.watch|vimeo\.com|reddit\.com|v\.redd\.it)\/[^\s]+/g;

    // Current video being edited
    let currentEditingVideo = null;

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => {
                c.classList.remove('active');
                c.classList.add('hidden');
            });

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            const content = document.getElementById(`${tabId}-tab`);
            content.classList.remove('hidden');
            content.classList.add('active');

            // Load library when switching to library tab
            if (tabId === 'library') {
                loadLibrary();
                loadFilters();
            }
        });
    });

    // Single Video Ingestion
    ingestBtn.addEventListener('click', async () => {
        const url = ingestInput.value.trim();
        if (!url) return;

        const topic = ingestTopic.value.trim();

        ingestStatus.textContent = 'Starting processing...';
        ingestStatus.className = 'status-msg processing';
        ingestStatus.classList.remove('hidden');
        ingestBtn.disabled = true;

        try {
            const response = await fetch('/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, topic })
            });

            const data = await response.json();

            if (response.ok) {
                ingestStatus.textContent = `Success: ${data.message}. It may take a minute to appear in search.`;
                ingestStatus.className = 'status-msg success';
                ingestInput.value = '';
                ingestTopic.value = '';
            } else {
                throw new Error(data.detail || 'Failed to start ingestion');
            }
        } catch (error) {
            ingestStatus.textContent = `Error: ${error.message}`;
            ingestStatus.className = 'status-msg error';
        } finally {
            ingestBtn.disabled = false;
        }
    });

    // Bulk Upload - URL counting
    bulkInput.addEventListener('input', () => {
        const text = bulkInput.value;
        const urls = extractUrls(text);
        urlCount.textContent = `${urls.length} URL${urls.length !== 1 ? 's' : ''} detected`;

        if (urls.length > 50) {
            urlCount.classList.add('error');
            urlCount.textContent += ' (max 50)';
        } else {
            urlCount.classList.remove('error');
        }
    });

    // Bulk Upload - Submit
    bulkBtn.addEventListener('click', async () => {
        const text = bulkInput.value;
        const urls = extractUrls(text);
        const topic = bulkTopic.value.trim();

        if (urls.length === 0) {
            bulkStatus.textContent = 'No valid TikTok URLs found';
            bulkStatus.className = 'status-msg error';
            bulkStatus.classList.remove('hidden');
            return;
        }

        if (urls.length > 50) {
            bulkStatus.textContent = 'Maximum 50 URLs per batch. Please reduce the number of URLs.';
            bulkStatus.className = 'status-msg error';
            bulkStatus.classList.remove('hidden');
            return;
        }

        bulkStatus.textContent = `Starting processing of ${urls.length} videos...`;
        bulkStatus.className = 'status-msg processing';
        bulkStatus.classList.remove('hidden');
        bulkBtn.disabled = true;

        try {
            const response = await fetch('/ingest/bulk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ urls, topic })
            });

            const data = await response.json();

            if (response.ok) {
                bulkStatus.textContent = `Success! ${data.count} videos queued for processing. They will appear in the library as they complete.`;
                bulkStatus.className = 'status-msg success';
                bulkInput.value = '';
                bulkTopic.value = '';
                urlCount.textContent = '0 URLs detected';
            } else {
                throw new Error(data.detail || 'Failed to start bulk ingestion');
            }
        } catch (error) {
            bulkStatus.textContent = `Error: ${error.message}`;
            bulkStatus.className = 'status-msg error';
        } finally {
            bulkBtn.disabled = false;
        }
    });

    // Helper function to extract video URLs
    function extractUrls(text) {
        const matches = text.match(VIDEO_URL_REGEX) || [];
        // Remove duplicates
        return [...new Set(matches)];
    }

    // Update min score display
    const minScoreSlider = document.getElementById('minScore');
    const minScoreValue = document.getElementById('minScoreValue');
    if (minScoreSlider && minScoreValue) {
        minScoreSlider.addEventListener('input', (e) => {
            minScoreValue.textContent = e.target.value + '%';
        });
    }

    // Allow Enter key to trigger search
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    searchBtn.addEventListener('click', performSearch);

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        const minScore = (document.getElementById('minScore')?.value || 0) / 100;
        const authorFilter = document.getElementById('authorFilter')?.value || '';

        loadingDiv.classList.remove('hidden');
        answerDiv.classList.add('hidden');
        matchesDiv.innerHTML = '';
        answerDiv.textContent = '';

        try {
            const url = new URL('/search', window.location.origin);
            url.searchParams.append('q', query);
            url.searchParams.append('min_score', minScore);
            if (authorFilter) {
                url.searchParams.append('author_filter', authorFilter);
            }

            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            answerDiv.textContent = `Error: ${error.message}`;
            answerDiv.classList.remove('hidden');
        } finally {
            loadingDiv.classList.add('hidden');
        }
    }

    function displayResults(data) {
        // First, create the source cards so we know how many there are
        const matches = data.matches || [];

        if (data.answer && matches.length > 0) {
            // Convert [Source X] citations to clickable links
            let answerHtml = data.answer
                .replace(/\n/g, '<br>')
                .replace(/\[Source (\d+)\]/g, (match, num) => {
                    const sourceNum = parseInt(num);
                    if (sourceNum <= matches.length) {
                        return `<a href="#source-${sourceNum}" class="source-link" onclick="scrollToSource(${sourceNum}); return false;">[Source ${sourceNum}]</a>`;
                    }
                    return match;
                });

            answerDiv.innerHTML = `<h3>AI Answer:</h3><p>${answerHtml}</p>`;
            answerDiv.classList.remove('hidden');
        } else if (data.answer) {
            answerDiv.innerHTML = `<h3>AI Answer:</h3><p>${data.answer.replace(/\n/g, '<br>')}</p>`;
            answerDiv.classList.remove('hidden');
        }

        if (matches.length > 0) {
            const matchesHeader = document.createElement('h3');
            matchesHeader.textContent = 'Source Videos:';
            matchesDiv.appendChild(matchesHeader);

            matches.forEach((match, index) => {
                const sourceNum = index + 1;
                const card = document.createElement('div');
                card.className = 'match-card';
                card.id = `source-${sourceNum}`;

                const score = (match.score * 100).toFixed(1);
                const metadata = match.metadata || {};
                const summary = metadata.summary || 'No summary available';
                const title = metadata.title || 'Untitled';
                const author = metadata.author || 'Unknown';
                const sourceUrl = metadata.source || '#';
                const topic = metadata.topic || '';
                const keyTakeaway = metadata.key_takeaway || '';
                const transcript = metadata.transcript || '';

                card.innerHTML = `
                    <div class="match-header">
                        <span class="source-number">[${sourceNum}]</span>
                        <span class="score">Match: ${score}%</span>
                        ${topic ? `<span class="topic-badge">${topic}</span>` : ''}
                        <a href="${sourceUrl}" target="_blank" class="video-link">Watch Video ↗</a>
                    </div>
                    <h4 class="video-title">${title}</h4>
                    <p class="video-author">By: ${author}</p>
                    ${keyTakeaway ? `<div class="key-takeaway"><span class="takeaway-label">Key Takeaway:</span> ${keyTakeaway}</div>` : ''}
                    <details class="content-details">
                        <summary>View Summary</summary>
                        <p class="match-summary">${summary}</p>
                    </details>
                    ${transcript ? `
                    <details class="content-details">
                        <summary>View Transcript</summary>
                        <p class="transcript-text">${transcript}</p>
                    </details>
                    ` : ''}
                `;
                matchesDiv.appendChild(card);
            });
        } else {
            matchesDiv.innerHTML = '<p>No matching videos found.</p>';
        }
    }

    // Global function to scroll to source
    window.scrollToSource = function(sourceNum) {
        const sourceCard = document.getElementById(`source-${sourceNum}`);
        if (sourceCard) {
            sourceCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // Highlight the card briefly
            sourceCard.classList.add('highlight');
            setTimeout(() => sourceCard.classList.remove('highlight'), 2000);
        }
    };

    // Library functions
    async function loadFilters() {
        try {
            const [topicsRes, authorsRes] = await Promise.all([
                fetch('/topics'),
                fetch('/authors')
            ]);

            const topicsData = await topicsRes.json();
            const authorsData = await authorsRes.json();

            // Populate topic filter
            topicFilter.innerHTML = '<option value="">All Topics</option>';
            topicsData.topics.forEach(topic => {
                const option = document.createElement('option');
                option.value = topic;
                option.textContent = topic;
                topicFilter.appendChild(option);
            });

            // Populate author filter
            authorFilterLibrary.innerHTML = '<option value="">All Creators</option>';
            authorsData.authors.forEach(author => {
                const option = document.createElement('option');
                option.value = author;
                option.textContent = author;
                authorFilterLibrary.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load filters:', error);
        }
    }

    async function loadLibrary() {
        libraryLoading.classList.remove('hidden');
        libraryVideos.innerHTML = '';
        categoryNav.innerHTML = '';

        const topic = topicFilter.value;
        const author = authorFilterLibrary.value;

        try {
            const url = new URL('/library', window.location.origin);
            if (topic) url.searchParams.append('topic', topic);
            if (author) url.searchParams.append('author', author);

            const response = await fetch(url);
            const data = await response.json();

            displayLibrary(data);
        } catch (error) {
            console.error('Failed to load library:', error);
            libraryVideos.innerHTML = `<p class="error">Failed to load library: ${error.message}</p>`;
        } finally {
            libraryLoading.classList.add('hidden');
        }
    }

    function displayLibrary(data) {
        const videos = data.videos || [];
        const sortMode = librarySortBy.value;

        // Show stats
        libraryStats.innerHTML = `<p>${videos.length} video${videos.length !== 1 ? 's' : ''} in your library</p>`;

        if (videos.length === 0) {
            libraryVideos.innerHTML = '<p class="empty-state">No videos found. Add some videos to get started!</p>';
            categoryNav.innerHTML = '';
            return;
        }

        // Helper to check if video is recent (within last 24 hours)
        function isRecent(video) {
            if (!video.ingested_at) return false;
            const now = Date.now() / 1000; // Convert to seconds
            const dayAgo = now - 86400;
            return video.ingested_at > dayAgo;
        }

        // Helper to create a video card
        function createVideoCard(video) {
            const card = document.createElement('div');
            card.className = 'library-card';

            const uploadDate = video.upload_date ? formatDate(video.upload_date) : '';
            const duration = video.duration ? formatDuration(video.duration) : '';
            const keyTakeaway = video.key_takeaway || '';
            const transcript = video.transcript || '';
            const summary = video.summary || '';
            const categories = video.categories ? video.categories.split(',').map(c => c.trim()) : [];
            const recentBadge = isRecent(video) ? '<span class="recent-badge">Recent</span>' : '';

            card.innerHTML = `
                <div class="library-card-header">
                    <h5 class="library-title">${recentBadge}${video.title || 'Untitled'}</h5>
                    <div class="card-actions">
                        <button class="reprocess-btn" title="Reprocess with AI">🔄</button>
                        <button class="edit-details-btn" title="Edit title, summary, takeaway">✏️</button>
                        <button class="edit-categories-btn" title="Edit categories">🏷️</button>
                        <a href="${video.source}" target="_blank" class="video-link">↗</a>
                    </div>
                </div>
                <p class="library-author">@${video.author || 'Unknown'}</p>
                <div class="library-meta">
                    ${uploadDate ? `<span>${uploadDate}</span>` : ''}
                    ${duration ? `<span>${duration}</span>` : ''}
                </div>
                ${categories.length > 1 ? `<div class="multi-category-badges">${categories.map(c => `<span class="small-badge">${c}</span>`).join('')}</div>` : ''}
                ${keyTakeaway ? `<div class="key-takeaway-small"><strong>Key:</strong> ${keyTakeaway}</div>` : ''}
                <details class="content-details">
                    <summary>Summary</summary>
                    <p class="library-summary">${summary}</p>
                </details>
                ${transcript ? `
                <details class="content-details">
                    <summary>Transcript</summary>
                    <p class="transcript-text">${transcript}</p>
                </details>
                ` : ''}
            `;

            // Add click handler for reprocess button
            const reprocessBtn = card.querySelector('.reprocess-btn');
            reprocessBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                reprocessBtn.disabled = true;
                reprocessBtn.textContent = '⏳';
                try {
                    const response = await fetch('/video/reprocess', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ video_id: video.id })
                    });
                    const result = await response.json();
                    if (response.ok) {
                        reprocessBtn.textContent = '✅';
                        setTimeout(() => loadLibrary(), 1000);
                    } else {
                        reprocessBtn.textContent = '❌';
                        alert('Reprocess failed: ' + (result.detail || 'Unknown error'));
                    }
                } catch (err) {
                    reprocessBtn.textContent = '❌';
                    alert('Reprocess failed: ' + err.message);
                }
                setTimeout(() => {
                    reprocessBtn.disabled = false;
                    reprocessBtn.textContent = '🔄';
                }, 2000);
            });

            // Add click handler for edit categories button
            const editCatBtn = card.querySelector('.edit-categories-btn');
            editCatBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                openVideoCategoryModal(video);
            });

            // Add click handler for edit details button
            const editDetailsBtn = card.querySelector('.edit-details-btn');
            editDetailsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                openVideoEditModal(video);
            });

            return card;
        }

        libraryVideos.innerHTML = '';

        if (sortMode === 'date') {
            // Sort by date - flat list, most recent first
            categoryNav.innerHTML = '';

            const sortedVideos = [...videos].sort((a, b) => (b.ingested_at || 0) - (a.ingested_at || 0));

            const videoGrid = document.createElement('div');
            videoGrid.className = 'video-grid';

            sortedVideos.forEach(video => {
                videoGrid.appendChild(createVideoCard(video));
            });

            libraryVideos.appendChild(videoGrid);
        } else {
            // Sort by category - grouped view
            const groupedByTopic = {};
            videos.forEach(video => {
                const categories = video.categories ? video.categories.split(',').map(c => c.trim()) : [video.topic || 'Uncategorized'];
                categories.forEach(cat => {
                    if (!cat) cat = 'Uncategorized';
                    if (!groupedByTopic[cat]) {
                        groupedByTopic[cat] = [];
                    }
                    if (!groupedByTopic[cat].find(v => v.id === video.id)) {
                        groupedByTopic[cat].push(video);
                    }
                });
            });

            const sortedTopics = Object.keys(groupedByTopic).sort();
            categoryNav.innerHTML = '<span class="nav-label">Jump to:</span> ' +
                sortedTopics.map(topic =>
                    `<a href="#category-${encodeURIComponent(topic)}" class="category-nav-link" onclick="scrollToCategory('${topic}'); return false;">${topic} (${groupedByTopic[topic].length})</a>`
                ).join(' ');

            sortedTopics.forEach(topic => {
                const section = document.createElement('div');
                section.className = 'library-section';
                section.id = `category-${encodeURIComponent(topic)}`;

                const header = document.createElement('h4');
                header.className = 'topic-header';
                header.innerHTML = `<span class="topic-name">${topic}</span> <span class="topic-count">(${groupedByTopic[topic].length})</span>`;
                section.appendChild(header);

                const videoGrid = document.createElement('div');
                videoGrid.className = 'video-grid';

                groupedByTopic[topic].forEach(video => {
                    videoGrid.appendChild(createVideoCard(video));
                });

                section.appendChild(videoGrid);
                libraryVideos.appendChild(section);
            });
        }
    }

    // Global function to scroll to category
    window.scrollToCategory = function(topic) {
        const section = document.getElementById(`category-${encodeURIComponent(topic)}`);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };

    // Helper functions
    function formatDate(dateStr) {
        if (!dateStr || dateStr.length !== 8) return '';
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        return `${month}/${day}/${year}`;
    }

    function formatDuration(seconds) {
        if (!seconds) return '';
        const totalSeconds = Math.round(seconds);
        const mins = Math.floor(totalSeconds / 60);
        const secs = totalSeconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function truncate(str, length) {
        if (!str) return '';
        if (str.length <= length) return str;
        return str.substring(0, length) + '...';
    }

    // Category Management Modal
    manageCategoriesBtn.addEventListener('click', () => {
        categoryModal.classList.remove('hidden');
        loadCategoryList();
    });

    closeCategoryModal.addEventListener('click', () => {
        categoryModal.classList.add('hidden');
    });

    addCategoryBtn.addEventListener('click', async () => {
        const name = newCategoryInput.value.trim();
        if (!name) return;

        try {
            const response = await fetch('/categories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });

            if (response.ok) {
                newCategoryInput.value = '';
                loadCategoryList();
                loadFilters();
            }
        } catch (error) {
            console.error('Failed to add category:', error);
        }
    });

    async function loadCategoryList() {
        try {
            const response = await fetch('/categories');
            const data = await response.json();

            categoryList.innerHTML = data.categories.map(cat => `
                <div class="category-item">
                    <span>${cat}</span>
                    <button class="delete-category-btn" data-category="${cat}">×</button>
                </div>
            `).join('');

            // Add delete handlers
            categoryList.querySelectorAll('.delete-category-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const catName = btn.getAttribute('data-category');
                    try {
                        await fetch(`/categories/${encodeURIComponent(catName)}`, { method: 'DELETE' });
                        loadCategoryList();
                        loadFilters();
                    } catch (error) {
                        console.error('Failed to delete category:', error);
                    }
                });
            });
        } catch (error) {
            console.error('Failed to load categories:', error);
        }
    }

    // Video Category Edit Modal
    async function openVideoCategoryModal(video) {
        currentEditingVideo = video;
        editVideoTitle.textContent = video.title || 'Untitled';

        try {
            const response = await fetch('/categories');
            const data = await response.json();
            const allCategories = data.categories || [];

            const currentCategories = video.categories ? video.categories.split(',').map(c => c.trim()) : [video.topic || ''];

            videoCategoryCheckboxes.innerHTML = allCategories.map(cat => `
                <label class="category-checkbox">
                    <input type="checkbox" value="${cat}" ${currentCategories.includes(cat) ? 'checked' : ''}>
                    ${cat}
                </label>
            `).join('');

            videoCategoryModal.classList.remove('hidden');
        } catch (error) {
            console.error('Failed to load categories for editing:', error);
        }
    }

    closeVideoCategoryModal.addEventListener('click', () => {
        videoCategoryModal.classList.add('hidden');
        currentEditingVideo = null;
    });

    saveVideoCategoriesBtn.addEventListener('click', async () => {
        if (!currentEditingVideo) return;

        const selectedCategories = Array.from(videoCategoryCheckboxes.querySelectorAll('input:checked'))
            .map(cb => cb.value);

        if (selectedCategories.length === 0) {
            alert('Please select at least one category');
            return;
        }

        try {
            const response = await fetch('/video/categories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: currentEditingVideo.id,
                    categories: selectedCategories
                })
            });

            if (response.ok) {
                videoCategoryModal.classList.add('hidden');
                currentEditingVideo = null;
                loadLibrary(); // Refresh library
            } else {
                const data = await response.json();
                alert(`Error: ${data.detail || 'Failed to update categories'}`);
            }
        } catch (error) {
            console.error('Failed to update video categories:', error);
            alert('Failed to update categories');
        }
    });

    // Video Edit Modal - Using div-based modal (not native dialog)
    // Native dialog has issues in VSCode webview with unexpected cancel events during text selection
    function openVideoEditModal(video) {
        currentEditingVideo = video;
        editTitleInput.value = video.title || '';
        editKeyTakeawayInput.value = video.key_takeaway || '';
        editSummaryInput.value = video.summary || '';
        videoEditModal.classList.remove('hidden');
    }

    closeVideoEditModal.addEventListener('click', () => {
        videoEditModal.classList.add('hidden');
        currentEditingVideo = null;
    });

    saveVideoEditBtn.addEventListener('click', async () => {
        if (!currentEditingVideo) return;

        const title = editTitleInput.value.trim();
        const keyTakeaway = editKeyTakeawayInput.value.trim();
        const summary = editSummaryInput.value.trim();

        if (!title) {
            alert('Title cannot be empty');
            return;
        }

        saveVideoEditBtn.disabled = true;
        saveVideoEditBtn.textContent = 'Saving...';

        try {
            const response = await fetch('/video/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: currentEditingVideo.id,
                    title: title,
                    summary: summary,
                    key_takeaway: keyTakeaway
                })
            });

            if (response.ok) {
                videoEditModal.classList.add('hidden');
                currentEditingVideo = null;
                loadLibrary(); // Refresh library
            } else {
                const data = await response.json();
                alert(`Error: ${data.detail || 'Failed to update video'}`);
            }
        } catch (error) {
            console.error('Failed to update video:', error);
            alert('Failed to update video');
        } finally {
            saveVideoEditBtn.disabled = false;
            saveVideoEditBtn.textContent = 'Save Changes';
        }
    });

    // Close modals only via the X button (no backdrop click)
    // This prevents accidental closures when selecting text in form inputs

    // Prevent any click/mouseup on modal backdrop from doing anything
    // The modal can ONLY be closed via the X button
    [categoryModal, videoCategoryModal, videoEditModal].forEach(modal => {
        modal.addEventListener('click', (e) => {
            // Only allow clicks to propagate if they're on actual interactive elements
            if (e.target === modal) {
                e.stopPropagation();
                e.preventDefault();
            }
        });
        modal.addEventListener('mouseup', (e) => {
            if (e.target === modal) {
                e.stopPropagation();
                e.preventDefault();
            }
        });
        modal.addEventListener('mousedown', (e) => {
            if (e.target === modal) {
                e.stopPropagation();
                e.preventDefault();
            }
        });
    });

    // Library filter listeners
    topicFilter.addEventListener('change', loadLibrary);
    authorFilterLibrary.addEventListener('change', loadLibrary);
    librarySortBy.addEventListener('change', loadLibrary);
    refreshLibrary.addEventListener('click', () => {
        loadFilters();
        loadLibrary();
    });
});
