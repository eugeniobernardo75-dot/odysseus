document.addEventListener('DOMContentLoaded', () => {

    /* =============================================
       Preloader
       ============================================= */
    const preloader = document.getElementById('preloader');
    window.addEventListener('load', () => {
        setTimeout(() => {
            preloader.classList.add('hidden');
        }, 500);
    });

    // Fallback: hide preloader after 3 seconds
    setTimeout(() => {
        preloader.classList.add('hidden');
    }, 3000);

    /* =============================================
       Navbar Scroll Effect
       ============================================= */
    const navbar = document.getElementById('navbar');
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');

    function handleNavbarScroll() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }

    function highlightActiveNav() {
        const scrollPos = window.scrollY + 100;
        sections.forEach(section => {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            const id = section.getAttribute('id');
            if (scrollPos >= top && scrollPos < top + height) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    window.addEventListener('scroll', () => {
        handleNavbarScroll();
        highlightActiveNav();
        handleBackToTop();
    });

    /* =============================================
       Mobile Menu
       ============================================= */
    const mobileToggle = document.getElementById('mobileToggle');
    const mobileOverlay = document.getElementById('mobileOverlay');
    const mobileNavLinks = document.querySelectorAll('.mobile-nav-link');
    const mobileNavCta = document.querySelector('.mobile-nav-cta');

    function toggleMobileMenu() {
        mobileToggle.classList.toggle('active');
        mobileOverlay.classList.toggle('active');
        document.body.style.overflow = mobileOverlay.classList.contains('active') ? 'hidden' : '';
    }

    function closeMobileMenu() {
        mobileToggle.classList.remove('active');
        mobileOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }

    mobileToggle.addEventListener('click', toggleMobileMenu);
    mobileOverlay.addEventListener('click', (e) => {
        if (e.target === mobileOverlay) closeMobileMenu();
    });

    mobileNavLinks.forEach(link => {
        link.addEventListener('click', closeMobileMenu);
    });

    if (mobileNavCta) {
        mobileNavCta.addEventListener('click', closeMobileMenu);
    }

    /* =============================================
       Smooth Scroll
       ============================================= */
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offsetTop = target.offsetTop - 80;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    /* =============================================
       Scroll Animations
       ============================================= */
    const animateElements = document.querySelectorAll('[data-animate]');

    function animateOnScroll() {
        animateElements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const delay = parseInt(el.dataset.delay) || 0;
            if (rect.top < window.innerHeight - 80) {
                setTimeout(() => {
                    el.classList.add('animated');
                }, delay);
            }
        });
    }

    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Run on load

    /* =============================================
       Counter Animation
       ============================================= */
    const numberValues = document.querySelectorAll('.number-value');
    let countersStarted = false;

    function startCounters() {
        if (countersStarted) return;

        const numbersSection = document.getElementById('numeros');
        if (!numbersSection) return;

        const rect = numbersSection.getBoundingClientRect();
        if (rect.top < window.innerHeight - 200) {
            countersStarted = true;

            numberValues.forEach(el => {
                const target = parseFloat(el.dataset.count);
                const decimals = parseInt(el.dataset.decimals) || 0;
                const prefix = el.dataset.prefix || '';
                const suffix = el.dataset.suffix || '';
                const duration = 2000;
                const startTime = performance.now();

                function updateCounter(currentTime) {
                    const elapsed = currentTime - startTime;
                    const progress = Math.min(elapsed / duration, 1);

                    // Ease out cubic
                    const eased = 1 - Math.pow(1 - progress, 3);
                    const current = target * eased;

                    el.textContent = prefix + current.toFixed(decimals) + suffix;

                    if (progress < 1) {
                        requestAnimationFrame(updateCounter);
                    } else {
                        el.textContent = prefix + target.toFixed(decimals) + suffix;
                    }
                }

                requestAnimationFrame(updateCounter);
            });
        }
    }

    window.addEventListener('scroll', startCounters);

    /* =============================================
       Portfolio Filter
       ============================================= */
    const filterBtns = document.querySelectorAll('.filter-btn');
    const portfolioCards = document.querySelectorAll('.portfolio-card');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const filter = btn.dataset.filter;

            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            portfolioCards.forEach(card => {
                if (filter === 'all' || card.dataset.category === filter) {
                    card.classList.remove('hidden');
                    card.style.animation = 'fadeIn 0.5s ease forwards';
                } else {
                    card.classList.add('hidden');
                }
            });
        });
    });

    /* =============================================
       Testimonials Slider
       ============================================= */
    const testimonialCards = document.querySelectorAll('.testimonial-card');
    const dots = document.querySelectorAll('.dot');
    let currentTestimonial = 0;
    let testimonialInterval;

    function showTestimonial(index) {
        testimonialCards.forEach(card => card.classList.remove('active'));
        dots.forEach(dot => dot.classList.remove('active'));

        testimonialCards[index].classList.add('active');
        dots[index].classList.add('active');
        currentTestimonial = index;
    }

    function nextTestimonial() {
        const next = (currentTestimonial + 1) % testimonialCards.length;
        showTestimonial(next);
    }

    dots.forEach(dot => {
        dot.addEventListener('click', () => {
            const index = parseInt(dot.dataset.index);
            showTestimonial(index);
            clearInterval(testimonialInterval);
            testimonialInterval = setInterval(nextTestimonial, 5000);
        });
    });

    // Auto-play
    testimonialInterval = setInterval(nextTestimonial, 5000);

    /* =============================================
       Contact Form
       ============================================= */
    const contactForm = document.getElementById('contactForm');
    const formSuccess = document.getElementById('formSuccess');

    if (contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();

            // Simulate form submission
            const submitBtn = contactForm.querySelector('button[type="submit"]');
            submitBtn.innerHTML = '<span>Enviando...</span> <i class="fas fa-spinner fa-spin"></i>';
            submitBtn.disabled = true;

            setTimeout(() => {
                contactForm.style.display = 'none';
                formSuccess.style.display = 'block';
            }, 1500);
        });
    }

    /* =============================================
       Back to Top
       ============================================= */
    const backToTop = document.getElementById('backToTop');

    function handleBackToTop() {
        if (window.scrollY > 500) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    }

    if (backToTop) {
        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    /* =============================================
       Hero Particles
       ============================================= */
    const particlesContainer = document.getElementById('heroParticles');

    function createParticle() {
        const particle = document.createElement('div');
        const size = Math.random() * 4 + 1;
        const x = Math.random() * 100;
        const duration = Math.random() * 10 + 10;
        const delay = Math.random() * 5;

        particle.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: rgba(108, 92, 231, ${Math.random() * 0.3 + 0.1});
            border-radius: 50%;
            left: ${x}%;
            bottom: -10px;
            animation: particleRise ${duration}s linear ${delay}s infinite;
            pointer-events: none;
        `;

        particlesContainer.appendChild(particle);
    }

    // Add particle animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes particleRise {
            0% { transform: translateY(0) translateX(0); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(-100vh) translateX(${Math.random() * 100 - 50}px); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Create particles
    for (let i = 0; i < 30; i++) {
        createParticle();
    }

    /* =============================================
       Phone Mask
       ============================================= */
    const phoneInput = document.getElementById('phone');
    if (phoneInput) {
        phoneInput.addEventListener('input', (e) => {
            let value = e.target.value.replace(/\D/g, '');
            if (value.length > 11) value = value.slice(0, 11);

            if (value.length > 6) {
                value = `(${value.slice(0, 2)}) ${value.slice(2, 7)}-${value.slice(7)}`;
            } else if (value.length > 2) {
                value = `(${value.slice(0, 2)}) ${value.slice(2)}`;
            } else if (value.length > 0) {
                value = `(${value}`;
            }

            e.target.value = value;
        });
    }

    /* =============================================
       Service Card Tilt Effect (Desktop)
       ============================================= */
    if (window.innerWidth > 768) {
        const serviceCards = document.querySelectorAll('.service-card');
        serviceCards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = (y - centerY) / 20;
                const rotateY = (centerX - x) / 20;
                card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = '';
            });
        });
    }

    /* =============================================
       Chatbot - Virtual Assistant
       ============================================= */
    const chatbot = {
        widget: document.getElementById('chatbotWidget'),
        toggle: document.getElementById('chatbotToggle'),
        window: document.getElementById('chatbotWindow'),
        messages: document.getElementById('chatbotMessages'),
        form: document.getElementById('chatbotForm'),
        input: document.getElementById('chatbotInput'),
        sendBtn: document.getElementById('chatbotSendBtn'),
        minimize: document.getElementById('chatbotMinimize'),
        quickReplies: document.getElementById('chatbotQuickReplies'),
        notification: document.querySelector('.chatbot-notification'),
        isOpen: false,
        conversationCount: 0
    };

    // ── Knowledge Base ──
    const knowledge = {
        saudacoes: [
            { match: /\b(ol[áa]|oi|hey|bom dia|boa tarde|boa noite|e a[ií]|fala a[ií]|iae?|beleza|tudo bem|opa|hello|hi)\b/i, reply: null }
        ],
        servicos: {
            match: /\b(servi[çc]os|o que faz|atua[çc][ãa]o|solu[çc][õo]es|o que oferecem|trabalham|ajuda|fazem|serviço|area|área)\b/i,
            reply: 'Oferecemos **6 áreas principais** de atuação:\n\n📌 **Estruturação Societária** — Holdings familiar, patrimonial e operacional\n📌 **Gestão de Investimentos** — Renda fixa/variável, private equity, venture capital\n📌 **Planejamento Tributário** — Redução legal de carga fiscal\n📌 **M&A e Fusões** — Compra, venda e fusão de empresas\n📌 **Consultoria Estratégica** — Business plans e crescimento acelerado\n📌 **Assessoria Jurídica** — Direito societário e governança corporativa\n\nQual área te interessa mais? 👇'
        },
        holding: {
            match: /\b(o que é holding|holding|estrutura[çc][ãa]o societ[áa]ria|holding familiar|holding patrimonial|holding operacional)\b/i,
            reply: 'Uma **holding** é uma empresa criada para controlar outras empresas (holdings). Principais benefícios:\n\n🏢 **Proteção Patrimonial** — Separa o patrimônio pessoal do empresarial\n💰 **Eficiência Tributária** — Redução significativa de impostos\n📋 **Planejamento Sucessório** — Transmissão organizada de bens\n⚖️ **Governança** — Gestão profissionalizada\n\nNa Vértice, somos especialistas em criar a estrutura ideal para cada caso!'
        },
        contato: {
            match: /\b(contato|telefone|whatsapp|email|endere[çc]o|falar com|falar|ligar|localiza[çc][ãa]o|onde fica|endereço|escrit[óo]rio)\b/i,
            reply: 'Você pode falar conosco pelos canais abaixo:\n\n📍 **Escritório Principal**\nAv. Paulista, 1578 - Sala 2201\nSão Paulo - SP, 01310-200\n\n📞 **Telefone:** (11) 3000-1234\n📱 **WhatsApp:** (11) 99999-8888\n📧 **E-mail:** contato@verteceholdings.com.br\n\n⏰ **Horário:** Seg-Sex: 9h às 18h | Sáb: 9h às 13h\n\nQuer que eu te ajude a agendar uma consultoria gratuita? 😊'
        },
        consultoria: {
            match: /\b(consultoria|consultar|diagn[óo]stico|an[áa]lise|estrat[ée]gia|business plan|planejamento|gratuita|gratuito|agendar|or[çc]amento|pre[çc]o|valor|investimento|quanto custa|preço)\b/i,
            reply: 'Oferecemos **consultoria gratuita** para entender suas necessidades! 🎯\n\nNa primeira reunião (online ou presencial):\n✅ Analisamos sua estrutura atual\n✅ Identificamos oportunidades de melhoria\n✅ Apresentamos um plano de ação personalizado\n\n👉 Para agendar, basta preencher o formulário de contato na seção **"Contato"** ou nos chamar no WhatsApp (11) 99999-8888.\n\nPosso te ajudar com mais informações sobre nossos serviços enquanto isso?'
        },
        portfolio: {
            match: /\b(portf[óo]lio|case|cases|resultados|exemplo|cliente|empresas que atendem|projetos|trabalhos)\b/i,
            reply: 'Temos cases de sucesso em diversas áreas:\n\n🏭 **Indústria** — Metalúrgica Alpha: +45% valorização em 18 meses\n💻 **Tecnologia** — TechBridge: captação de R$50M, +180% receita\n🌿 **Agronegócio** — AgroVida: +60% patrimônio em 12 meses\n🏥 **Saúde** — MedPrime: fusão de 3 clínicas, +200% lucro em 30 meses\n\nQuer saber mais detalhes de algum case específico? 👇'
        },
        equipe: {
            match: /\b(equipe|time|quem s[ãa]o|lideran[çc]a|fundador|s[óo]cios|diretores|gestores|profissionais)\b/i,
            reply: 'Nossa liderança é composta por profissionais de alto nível:\n\n👤 **Carlos Eduardo Mendes** — CEO & Fundador (MBA FGV, 20+ anos)\n👤 **Ana Paula Ferreira** — CFO (CFA Charterholder)\n👤 **Roberto Nascimento** — Diretor Jurídico (Tributarista)\n👤 **Mariana Costa Lima** — Diretora de Operações (50+ operações M&A)\n\nSomos mais de 50 profissionais dedicados à sua holding!'
        },
        numeros: {
            match: /\b(n[úu]meros|resultados|estat[íi]sticas|indicadores|dados|faturamento|ativos|retorno|satisfa[çc][ãa]o)\b/i,
            reply: 'Nossos números comprovam nossa excelência 📊\n\n🏢 **247** empresas em gestão\n💰 **R$ 8,2 Bi** em ativos sob gestão\n📈 **18,7%** retorno médio acima do CDI\n❤️ **98%** de satisfação dos clientes\n\nSão mais de **15 anos** de mercado e crescimento contínuo!'
        },
        investimentos: {
            match: /\b(investimento|investir|aplica[çc][ãa]o|renda fixa|renda vari[áa]vel|fundo|a[çc][ãa]o|cdi|poupan[çc]a|tesouro|private equity|venture capital)\b/i,
            reply: 'Nossa **Gestão de Investimentos** abrange:\n\n📊 **Renda Fixa** — CDB, LCI, LCA, Tesouro Direto, Debêntures\n📈 **Renda Variável** — Ações, FIIs, ETFs, BDRs\n🏦 **Fundos** — Multimercado, Ações, Previdência\n💎 **Private Equity** — Participação em empresas promissoras\n🚀 **Venture Capital** — Startups com alto potencial\n\nTudo personalizado conforme seu perfil de risco e objetivos!'
        },
        tributario: {
            match: /\b(tribut[áa]rio|imposto|taxa|fiscal|receita|dividendo|lucro|presumido|real|simples|ir|irpf|irpj|csll|pis|cofins|icms|iss|carga tribut[áa]ria)\b/i,
            reply: 'Em **Planejamento Tributário**, atuamos com:\n\n✅ Distribuição estratégica de dividendos\n✅ Escolha do regime ideal (Lucro Presumido vs Real)\n✅ Incentivos fiscais\n✅ Compliance tributário completo\n✅ Redução legal da carga tributária\n\nNossos clientes economizam em média **32%** em impostos! 📉'
        },
        ma: {
            match: /\b(m&a|fus[ãa]o|aquisi[çc][ãa]o|due diligence|valuation|compra|venda|merge|incorpora[çc][ãa]o|cis[ãa]o)\b/i,
            reply: 'Nossa assessoria em **M&A e Fusões** inclui:\n\n🔍 **Due Diligence** completa\n💰 **Valuation** de empresas\n🤝 **Negociação Estratégica**\n🔄 **Integração Pós-Operação**\n\nJá conduzimos **50+ operações** com sucesso! Quer saber mais sobre como podemos ajudar na sua?'
        },
        juridico: {
            match: /\b(jur[ií]dico|advogado|contrato|societ[áa]rio|acordo|quotista|conflito|media[çc][ãa]o|lit[ií]gio|compliance|governan[çc]a)\b/i,
            reply: 'Nossa **Assessoria Jurídica** oferece:\n\n📝 Contratos societários personalizados\n🤝 Acordos de quotistas e acionistas\n🏛️ Governança corporativa\n⚖️ Mediação de conflitos entre sócios\n📋 Compliance e adequação regulatória\n\nConte com nosso diretor jurídico, Dr. Roberto Nascimento, para toda segurança legal!'
        },
        duvida: {
            match: /\b(d[vú]vida|n[ãa]o entendi|ajuda|socorro|pode repetir|como funciona|explica|entender)\b/i,
            reply: 'Claro! Vou te ajudar 😊\n\nA **Grupo Vértice Holdings** é uma empresa de gestão empresarial e investimentos, especializada em estruturar e administrar holdings.\n\nFundada em 2010, já atendemos mais de **200 empresas** em todo o Brasil, gerenciando **R$ 8,2 bilhões** em ativos.\n\nQue informação você gostaria de saber? Posso te falar sobre:\n🏢 Serviços | 💰 Investimentos | 📞 Contato | 🎯 Consultoria'
        },
        obrigado: {
            match: /\b(obrigado|obrigada|valeu|brigado|thanks|thank|agrade[çc]o|grato|show|top|demais|ótimo|perfeito|excelente)\b/i,
            reply: 'Por nada! 😊 Fico feliz em ajudar!\n\nSe tiver mais alguma dúvida, é só perguntar.\n\nAh, e não esqueça de agendar sua **consultoria gratuita**! 🎯 Basta preencher o formulário na seção de Contato ou nos chamar no WhatsApp (11) 99999-8888.'
        },
        preco: {
            match: /\b(quanto custa|pre[çc]o|valor|investimento|or[çc]amento|mensalidade|taxa|honor[áa]rio|custa|caro|barato)\b/i,
            reply: 'Ótima pergunta! 💰\n\nO investimento varia conforme a complexidade de cada caso. Por isso, oferecemos **consultoria gratuita** inicial!\n\nNela, analisamos:\n✅ Sua estrutura atual\n✅ Seus objetivos\n✅ O melhor plano de ação\n✅ E apresentamos uma proposta personalizada\n\n👉 **Agende agora mesmo** pelo formulário de contato ou WhatsApp (11) 99999-8888!'
        },
        tempo: {
            match: /\b(quanto tempo|prazo|demora|r[áa]pido|urg[êe]ncia|prioridade|quando fica pronto|agendar|marcar)\b/i,
            reply: 'Os prazos variam conforme o tipo de serviço:\n\n⏱️ **Consultoria inicial** — agendamos em até 48h\n📋 **Diagnóstico** — 1 a 2 semanas\n🏗️ **Estruturação de Holding** — 30 a 60 dias\n💼 **M&A** — 3 a 6 meses (depende da complexidade)\n\nQuer saber o prazo para o seu caso específico? Agende uma consultoria gratuita!'
        },
        fallback: [
            'Olá! 😊 Posso ajudar com informações sobre nossos serviços.\n\n📋 **Serviços** — O que oferecemos\n🏢 **O que é Holding** — Entenda o conceito\n📞 **Contato** — Nossos canais\n🎯 **Consultoria** — Agende uma reunião gratuita',
            'Entendi! Deixe-me ver se posso ajudar melhor 😊\n\nTente perguntar sobre:\n• **Serviços** que oferecemos\n• **Investimentos**\n• **Contato** e endereço\n• **Consultoria gratuita**\n• Nossos **cases de sucesso**',
            'Que legal seu interesse! Vamos lá 👇\n\nPosso te explicar sobre:\n📌 **Estruturação de Holdings**\n📌 **Gestão de Investimentos**\n📌 **Planejamento Tributário**\n📌 **M&A e Fusões**\n📌 **Consultoria Estratégica**\n📌 **Assessoria Jurídica**\n\nQual te interessa mais?'
        ]
    };

    // ── Helper Functions ──
    function getCurrentTime() {
        const now = new Date();
        return now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0');
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatbot.messages.scrollTop = chatbot.messages.scrollHeight;
        }, 50);
    }

    function addMessage(text, type) {
        const div = document.createElement('div');
        div.className = `chatbot-message ${type}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'chatbot-message-avatar';
        avatar.innerHTML = type === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        const bubble = document.createElement('div');
        bubble.className = 'chatbot-message-bubble';
        bubble.innerHTML = text.replace(/\n/g, '<br>') + '<span class="chatbot-message-time">' + getCurrentTime() + '</span>';
        
        div.appendChild(avatar);
        div.appendChild(bubble);
        chatbot.messages.appendChild(div);
        scrollToBottom();
    }

    function addTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'chatbot-typing';
        div.id = 'chatbotTyping';
        
        const avatar = document.createElement('div');
        avatar.className = 'chatbot-message-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        
        const bubbles = document.createElement('div');
        bubbles.className = 'chatbot-typing-bubbles';
        bubbles.innerHTML = '<span></span><span></span><span></span>';
        
        div.appendChild(avatar);
        div.appendChild(bubbles);
        chatbot.messages.appendChild(div);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const typing = document.getElementById('chatbotTyping');
        if (typing) typing.remove();
    }

    function getResponse(userMessage) {
        const msg = userMessage.toLowerCase().trim();
        
        // Check specific intents
        for (const [key, data] of Object.entries(knowledge)) {
            if (key === 'saudacoes' || key === 'fallback') continue;
            if (data.match && data.match.test(msg)) {
                return data.reply;
            }
        }

        // Check investment keywords
        if (/\b(investimento|investir|aplicação|renda|fundo|ação|cdi|poupança|tesouro|private equity|venture capital)\b/i.test(msg)) {
            return knowledge.investimentos.reply;
        }

        // Check greetings with context
        if (/\b(ol[áa]|oi|hey|bom dia|boa tarde|boa noite|e a[ií]|fala a[ií]|iae?|beleza|tudo bem|opa|hello|hi)\b/i.test(msg)) {
            if (chatbot.conversationCount < 2) {
                return 'Olá! 👋 Seja bem-vindo à **Grupo Vértice Holdings**!\n\nSou o assistente virtual e estou aqui para ajudar!\n\n📋 Quer conhecer nossos **Serviços**?\n🏢 Saber **o que é uma Holding**?\n📞 Falar com **nossa equipe**?\n🎯 Agendar uma **Consultoria Gratuita**?\n\nÉ só perguntar! 😊';
            }
        }

        // Check what/which questions
        if (/\b(o que|qual|quais|como|quem|onde|quando|por que)\b/i.test(msg)) {
            return knowledge.duvida.reply;
        }

        // Check certificate/complaint words
        if (/\b(reclama[çc][ãa]o|problema|erro|n[ãa]o funcionou|péssimo|ruim)\b/i.test(msg)) {
            return 'Sinto muito pela experiência negativa! 😟\n\nPor favor, entre em contato diretamente com nossa equipe pelo **WhatsApp (11) 99999-8888** ou **e-mail: contato@verteceholdings.com.br** para que possamos resolver rapidamente.\n\nSeu feedback é muito importante para nós! 🙏';
        }

        // Random fallback
        const fallbacks = knowledge.fallback;
        return fallbacks[Math.floor(Math.random() * fallbacks.length)];
    }

    function simulateResponse(userMessage) {
        addTypingIndicator();
        chatbot.conversationCount++;
        
        const delay = Math.random() * 800 + 600;
        
        setTimeout(() => {
            removeTypingIndicator();
            const response = getResponse(userMessage);
            addMessage(response, 'bot');
        }, delay);
    }

    function handleSend() {
        const message = chatbot.input.value.trim();
        if (!message) return;
        
        chatbot.input.value = '';
        chatbot.sendBtn.disabled = true;
        
        // Hide quick replies
        chatbot.quickReplies.classList.add('hidden');
        
        // Add user message
        addMessage(escapeHtml(message), 'user');
        
        // Simulate bot response
        simulateResponse(message);
        
        setTimeout(() => {
            chatbot.sendBtn.disabled = false;
            chatbot.input.focus();
        }, 200);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function handleQuickReply(query) {
        // Hide quick replies
        chatbot.quickReplies.classList.add('hidden');
        
        // Show user clicked quick reply
        const labels = {
            'servicos': '📋 Quero saber os serviços',
            'holding': '🏢 O que é Holding?',
            'contato': '📞 Informações de contato',
            'consultoria': '🎯 Quero uma consultoria'
        };
        
        addMessage(labels[query] || query, 'user');
        chatbot.conversationCount++;
        
        // Show bot response
        const responses = {
            'servicos': knowledge.servicos.reply,
            'holding': knowledge.holding.reply,
            'contato': knowledge.contato.reply,
            'consultoria': knowledge.consultoria.reply
        };
        
        addTypingIndicator();
        setTimeout(() => {
            removeTypingIndicator();
            addMessage(responses[query], 'bot');
        }, Math.random() * 500 + 500);
    }

    // ── Event Listeners ──
    chatbot.toggle.addEventListener('click', () => {
        chatbot.isOpen = !chatbot.isOpen;
        chatbot.widget.classList.toggle('active');
        
        if (chatbot.isOpen) {
            chatbot.notification.classList.remove('visible');
            chatbot.input.focus();
            scrollToBottom();
        }
    });

    chatbot.minimize.addEventListener('click', () => {
        chatbot.isOpen = false;
        chatbot.widget.classList.remove('active');
    });

    chatbot.form.addEventListener('submit', (e) => {
        e.preventDefault();
        handleSend();
    });

    chatbot.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    chatbot.input.addEventListener('input', () => {
        chatbot.sendBtn.disabled = !chatbot.input.value.trim();
    });

    // Quick reply buttons
    document.querySelectorAll('.quick-reply-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            handleQuickReply(btn.dataset.query);
        });
    });

    // ── Notification Badge ──
    setTimeout(() => {
        if (!chatbot.isOpen) {
            chatbot.notification.classList.add('visible');
        }
    }, 5000);

    // ── Auto-greeting on page load (after 10s if not opened) ──
    setTimeout(() => {
        if (!chatbot.isOpen && chatbot.conversationCount === 0) {
            chatbot.notification.classList.add('visible');
        }
    }, 8000);

    /* =============================================
       Cursor Glow Effect (Desktop)
       ============================================= */
    if (window.innerWidth > 1024) {
        const hero = document.querySelector('.hero');
        if (hero) {
            const glow = document.createElement('div');
            glow.style.cssText = `
                position: absolute;
                width: 400px;
                height: 400px;
                background: radial-gradient(circle, rgba(108, 92, 231, 0.08) 0%, transparent 70%);
                border-radius: 50%;
                pointer-events: none;
                z-index: 1;
                transition: transform 0.3s ease;
                transform: translate(-50%, -50%);
            `;
            hero.appendChild(glow);

            hero.addEventListener('mousemove', (e) => {
                const rect = hero.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                glow.style.left = x + 'px';
                glow.style.top = y + 'px';
            });
        }
    }

});
