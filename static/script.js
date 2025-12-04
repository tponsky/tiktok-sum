document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const loadingDiv = document.getElementById('loading');
    const answerDiv = document.getElementById('answer');
    const matchesDiv = document.getElementById('matches');

    const ingestInput = document.getElementById('ingestInput');
    const ingestBtn = document.getElementById('ingestBtn');
    const ingestStatus = document.getElementById('ingestStatus');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => {
                c.classList.remove('active');
                c.classList.add('hidden');
            });

            // Add active class to clicked
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            const content = document.getElementById(`${tabId}-tab`);
            content.classList.remove('hidden');
            content.classList.add('active');
        });
    });

    // Ingestion Logic
    ingestBtn.addEventListener('click', async () => {
        const url = ingestInput.value.trim();
        if (!url) return;

        ingestStatus.textContent = 'Starting processing...';
        ingestStatus.className = 'status-msg processing';
        ingestStatus.classList.remove('hidden');
        ingestBtn.disabled = true;

        try {
            const response = await fetch('/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });

            const data = await response.json();

            if (response.ok) {
                ingestStatus.textContent = `Success: ${data.message}. It may take a minute to appear in search.`;
                ingestStatus.className = 'status-msg success';
                ingestInput.value = '';
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

        // Get filter values
        const minScore = (document.getElementById('minScore')?.value || 0) / 100;
        const authorFilter = document.getElementById('authorFilter')?.value || '';

        // Reset UI
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
        // Display Answer
        if (data.answer) {
            answerDiv.innerHTML = `<h3>AI Answer:</h3><p>${data.answer.replace(/\n/g, '<br>')}</p>`;
            answerDiv.classList.remove('hidden');
        }

        // Display Matches
        if (data.matches && data.matches.length > 0) {
            const matchesHeader = document.createElement('h3');
            matchesHeader.textContent = 'Source Videos:';
            matchesDiv.appendChild(matchesHeader);

            data.matches.forEach(match => {
                const card = document.createElement('div');
                card.className = 'match-card';

                const score = (match.score * 100).toFixed(1);
                const metadata = match.metadata || {};
                const summary = metadata.summary || 'No summary available';
                const title = metadata.title || 'Untitled';
                const author = metadata.author || 'Unknown';
                const sourceUrl = metadata.source || '#';

                card.innerHTML = `
                    <div class="match-header">
                        <span class="score">Match: ${score}%</span>
                        <a href="${sourceUrl}" target="_blank" class="video-link">Watch Video ↗</a>
                    </div>
                    <h4 class="video-title">${title}</h4>
                    <p class="video-author">By: ${author}</p>
                    <p class="match-summary">${summary}</p>
                `;
                matchesDiv.appendChild(card);
            });
        } else {
            matchesDiv.innerHTML = '<p>No matching videos found.</p>';
        }
    }
});
