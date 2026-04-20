window.HELP_IMPROVE_VIDEOJS = false;

const translations = {
    en: {
        pageTitle: 'ES4R | Academic Project Page',
        metaDescription: 'Project page for ES4R, an ACL 2026 paper on speech-based empathetic response generation with prepositive affective modeling, cross-modal fusion, and expressive speech synthesis.',
        metaShortDescription: 'Speech encoding based on prepositive affective modeling for empathetic response generation.',
        languageMeta: 'English',
        paperTitle: '<strong>ES4R:</strong><br><span style="font-weight:normal; font-size:0.75em;">Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation</span>',
        institutionLine: 'School of Computer Science and Engineering, Northeastern University, Shenyang 110819, China',
        venueLine: 'Annual Meeting of the Association for Computational Linguistics (ACL) 2026',
        authorNote: '<sup>†</sup>Co-first authors &nbsp; <sup>*</sup>Corresponding Author',
        codeButton: 'Code',
        paperButton: 'arXiv',
        abstractTitle: 'Abstract',
        abstractBody: 'Empathetic speech dialogue requires not only understanding linguistic content but also perceiving rich paralinguistic information such as prosody, tone, and emotional intensity for affective understanding. Existing speech-to-speech large language models either rely on ASR transcription or use encoders to extract latent representations, which often weakens affective information and contextual coherence in multi-turn dialogues. ES4R addresses this problem by explicitly modeling structured affective context before speech encoding. It introduces a dual-level attention mechanism to capture turn-level affective states and dialogue-level affective dynamics, then integrates these representations with textual semantics through speech-guided cross-modal attention to generate empathetic responses. For speech output, the framework further adopts energy-based strategy selection and style fusion for empathetic speech synthesis. ES4R consistently outperforms strong baselines in both automatic and human evaluations and remains robust across different large language model backbones.',
        architectureTitle: 'Architecture',
        architectureCaption: 'The overall framework of ES4R. Intra-turn and inter-turn attention support prepositive affective modeling, while speech-guided cross-modal fusion and dual-path training drive empathetic response generation.',
        analysesTitle: 'Analyses',
        analysis1: '<strong>Representation analysis</strong> of turn-level cosine similarity. The left and middle heatmaps show that ES4R\'s <strong>Dual-Attn module</strong> produces clearer <strong>intra-dialogue structure</strong> than the Whisper baseline. The right panel further shows that ES4R improves the <strong>dialogue cohesion ratio from 1.00× to 1.08×</strong>, validating the effectiveness of <strong>prepositive affective modeling</strong>.',
        analysis2: '<strong>Automatic evaluation results</strong> on the <strong>AvaMERG dataset</strong>. <strong>ES4R(Qwen)</strong> and <strong>ES4R(LLaMA)</strong> deliver the best or second-best scores across <strong>BLEU, ROUGE, and BERTScore</strong> against all baselines. <strong>Bold</strong> marks the best result and <u>underline</u> marks the runner-up. <strong>w/o Dual-Attn</strong> and <strong>w/o Cross-Attn</strong> denote ablation variants without dual-level attention or speech-guided cross-modal fusion.',
        analysis3: '<strong>Human evaluation results</strong> (A/B testing) on <strong>text responses</strong> across five criteria: <strong>Topic Understanding</strong>, <strong>Emotion Recognition</strong>, <strong>Response Specificity</strong>, <strong>Actionable Advice</strong>, and <strong>Empathy Depth</strong>. ES4R achieves consistently higher <strong>win rates</strong> across all dimensions and all comparisons.',
        analysis4: '<strong>Human evaluation results</strong> on dialogue-level <strong>speech responses</strong> using <strong>DMOS metrics</strong>. ES4R shows stronger <strong>Empathy Expressiveness (DMOS-E)</strong> and <strong>Emotional Consistency (DMOS-C)</strong> than OpenS2S and LLaMA-Omni 2, while maintaining competitive <strong>Speech Quality (DMOS-Q)</strong>.',
        analysis5: '<strong>LLM-based evaluation results</strong> using <strong>GPT-5</strong> across four dimensions: <strong>Quality</strong>, <strong>Empathy</strong>, <strong>Completeness</strong>, and <strong>Fluency</strong> (0-10). ES4R consistently outperforms baselines and ablations, reinforcing the value of <strong>prepositive affective modeling</strong> and <strong>cross-modal fusion</strong>.',
        demoTitle: 'Demo Examples',
        example1Title: 'Example 1',
        example2Title: 'Example 2',
        example3Title: 'Example 3',
        tableTurn: 'Turn',
        tableAudioInput: 'Audio Input',
        tableTranscript: 'Transcript',
        tableAudioOutput: 'Audio Output',
        tableTextOutput: 'Text Output',
        responseBadge: 'Response',
        scrollTopLabel: 'Scroll to top'
    },
    zh: {
        pageTitle: 'ES4R | 项目主页',
        metaDescription: 'ES4R 项目主页，展示 ACL 2026 关于语音共情回复生成的研究工作，重点包括前置情感建模、跨模态融合与表达性语音合成。',
        metaShortDescription: '基于前置情感建模的语音编码用于共情回复生成。',
        languageMeta: 'Chinese',
        paperTitle: '<strong>ES4R:</strong><br><span style="font-weight:normal; font-size:0.75em;">基于前置情感建模的语音编码用于共情回复生成</span>',
        institutionLine: '东北大学计算机科学与工程学院，中国沈阳 110819',
        venueLine: '计算语言学协会年会（ACL 2026）',
        authorNote: '<sup>†</sup>共同一作 &nbsp; <sup>*</sup>通讯作者',
        codeButton: '代码',
        paperButton: '论文',
        abstractTitle: '摘要',
        abstractBody: '语音共情对话不仅要求模型理解文本语义，还要求它感知韵律、语调和情绪强度等丰富的副语言信息。现有语音到语音大模型通常依赖 ASR 转写，或者仅用编码器抽取潜表示，这往往会削弱多轮对话中的情感信息与上下文连贯性。ES4R 通过在语音编码前显式建模结构化情感上下文来解决这一问题。具体而言，模型引入双层注意力机制以捕获轮次级情感状态和对话级情感动态，再通过语音引导的跨模态注意力将这些表示与文本语义融合，从而生成更具共情性的回复。在语音输出侧，框架进一步采用基于能量的策略选择和风格融合以支持共情语音合成。ES4R 在自动评测和人工评测中均持续优于强基线，并且在不同大语言模型骨干上保持稳定表现。',
        architectureTitle: '方法架构',
        architectureCaption: 'ES4R 的整体框架。轮内与轮间注意力共同支持前置情感建模，语音引导的跨模态融合与双路径训练共同驱动共情回复生成。',
        analysesTitle: '结果分析',
        analysis1: '<strong>表示分析</strong>展示了轮次级余弦相似度。左侧和中间热力图表明，ES4R 的 <strong>Dual-Attn 模块</strong>相比 Whisper 基线能够形成更清晰的<strong>对话内结构</strong>。右侧定量结果进一步显示，ES4R 将<strong>对话凝聚比从 1.00× 提升到 1.08×</strong>，验证了<strong>前置情感建模</strong>的有效性。',
        analysis2: '<strong>AvaMERG 数据集</strong>上的<strong>自动评测结果</strong>。<strong>ES4R(Qwen)</strong> 与 <strong>ES4R(LLaMA)</strong> 在 <strong>BLEU、ROUGE 和 BERTScore</strong> 等指标上取得了最佳或次优表现。<strong>加粗</strong>表示最优结果，<u>下划线</u>表示次优结果。<strong>w/o Dual-Attn</strong> 与 <strong>w/o Cross-Attn</strong> 分别表示移除双层注意力模块和语音引导跨模态融合模块的消融版本。',
        analysis3: '<strong>人工评测结果</strong>（A/B 测试）比较了 <strong>文本回复</strong> 在五个维度上的表现：<strong>主题理解</strong>、<strong>情绪识别</strong>、<strong>回复具体性</strong>、<strong>建议可执行性</strong> 与 <strong>共情深度</strong>。ES4R 在所有维度和所有比较中都取得了更高的<strong>胜率</strong>。',
        analysis4: '<strong>对话级语音回复</strong>上的<strong>人工评测结果</strong>采用了 <strong>DMOS 指标</strong>。与 OpenS2S 和 LLaMA-Omni 2 相比，ES4R 在 <strong>共情表达性（DMOS-E）</strong> 和 <strong>情绪一致性（DMOS-C）</strong> 上表现更强，同时保持了具有竞争力的 <strong>语音质量（DMOS-Q）</strong>。',
        analysis5: '基于 <strong>GPT-5</strong> 的<strong>大模型评测结果</strong>覆盖 <strong>质量</strong>、<strong>共情性</strong>、<strong>完整性</strong> 和 <strong>流畅度</strong> 四个维度（0-10 分）。ES4R 持续优于基线与消融模型，进一步验证了<strong>前置情感建模</strong>与<strong>跨模态融合</strong>的价值。',
        demoTitle: '示例案例',
        example1Title: '示例 1',
        example2Title: '示例 2',
        example3Title: '示例 3',
        tableTurn: '轮次',
        tableAudioInput: '输入音频',
        tableTranscript: '转写文本',
        tableAudioOutput: '输出音频',
        tableTextOutput: '输出文本',
        responseBadge: '模型回复',
        scrollTopLabel: '回到顶部'
    }
};

function applyTranslations(lang) {
    const locale = translations[lang] || translations.en;

    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = locale.pageTitle;

    const metaDescription = document.getElementById('meta-description');
    const metaOgDescription = document.getElementById('meta-og-description');
    const metaTwitterDescription = document.getElementById('meta-twitter-description');
    const metaLanguage = document.getElementById('meta-language');
    const scrollButton = document.getElementById('scrollToTopButton');

    if (metaDescription) metaDescription.setAttribute('content', locale.metaDescription);
    if (metaOgDescription) metaOgDescription.setAttribute('content', locale.metaShortDescription);
    if (metaTwitterDescription) metaTwitterDescription.setAttribute('content', locale.metaShortDescription);
    if (metaLanguage) metaLanguage.setAttribute('content', locale.languageMeta);

    document.querySelectorAll('[data-i18n]').forEach((element) => {
        const key = element.getAttribute('data-i18n');
        if (locale[key] !== undefined) {
            element.textContent = locale[key];
        }
    });

    document.querySelectorAll('[data-i18n-html]').forEach((element) => {
        const key = element.getAttribute('data-i18n-html');
        if (locale[key] !== undefined) {
            element.innerHTML = locale[key];
        }
    });

    if (scrollButton) {
        scrollButton.title = locale.scrollTopLabel;
        scrollButton.setAttribute('aria-label', locale.scrollTopLabel);
    }

    document.querySelectorAll('[data-lang-switch]').forEach((button) => {
        button.classList.toggle('is-active', button.getAttribute('data-lang-switch') === lang);
    });
}

function setLanguage(lang) {
    const normalizedLang = lang === 'zh' ? 'zh' : 'en';
    localStorage.setItem('es4r-language', normalizedLang);
    applyTranslations(normalizedLang);
}

function initializeLanguage() {
    const stored = localStorage.getItem('es4r-language');
    const browserPrefersZh = navigator.language && navigator.language.toLowerCase().startsWith('zh');
    const initialLang = stored || (browserPrefersZh ? 'zh' : 'en');
    applyTranslations(initialLang);

    document.querySelectorAll('[data-lang-switch]').forEach((button) => {
        button.addEventListener('click', function() {
            setLanguage(this.getAttribute('data-lang-switch'));
        });
    });
}

function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (!dropdown || !button) {
        return;
    }

    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (container && dropdown && button && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');

        if (dropdown && button) {
            dropdown.classList.remove('show');
            button.classList.remove('active');
        }
    }
});

function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');

    if (!bibtexElement || !button) {
        return;
    }

    const copyText = button.querySelector('.copy-text');

    navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
        button.classList.add('copied');
        if (copyText) copyText.textContent = 'Cop';

        setTimeout(function() {
            button.classList.remove('copied');
            if (copyText) copyText.textContent = 'Copy';
        }, 2000);
    }).catch(function() {
        const textArea = document.createElement('textarea');
        textArea.value = bibtexElement.textContent;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);

        button.classList.add('copied');
        if (copyText) copyText.textContent = 'Cop';
        setTimeout(function() {
            button.classList.remove('copied');
            if (copyText) copyText.textContent = 'Copy';
        }, 2000);
    });
}

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');

    if (!scrollButton) {
        return;
    }

    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');

    if (carouselVideos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                video.play().catch(() => {});
            } else {
                video.pause();
            }
        });
    }, {
        threshold: 0.5
    });

    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    initializeLanguage();

    const options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000
    };

    bulmaCarousel.attach('.carousel', options);
    bulmaSlider.attach();
    setupVideoCarouselAutoplay();
});
