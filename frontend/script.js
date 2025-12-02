// Carregar unidades do backend
async function carregarUnidades() {
  const select = document.getElementById('unidadeSelect');
  try {
    const response = await fetch('/unidades');
    const unidades = await response.json();
    select.innerHTML = '<option value="">Selecione a unidade</option>';
    unidades.forEach(u => {
      const opt = document.createElement('option');
      opt.value = u;
      opt.textContent = u;
      select.appendChild(opt);
    });
  } catch (err) {
    select.innerHTML = '<option value="">Erro ao carregar unidades</option>';
    console.error(err);
  }
}

carregarUnidades();

// Botão de previsão
document.getElementById('submitBtn').addEventListener('click', async () => {
  const dataSintoma = new Date(document.getElementById('dataSintoma').value);
  const dataNotificacao = new Date(document.getElementById('dataNotificacao').value);
  const dataInvestigacao = new Date(document.getElementById('dataInvestigacao').value);
  const unidade = document.getElementById('unidadeSelect').value;

  if (!dataSintoma || !dataNotificacao || !dataInvestigacao || !unidade) {
    alert("Preencha todas as datas e selecione a unidade!");
    return;
  }

  // Calcular diferenças em dias
  const tempoSinPriNotific = (dataNotificacao - dataSintoma) / (1000 * 60 * 60 * 24);
  const tempoInvestEncerrar = (dataInvestigacao - dataNotificacao) / (1000 * 60 * 60 * 24);

  // Enviar para o backend
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tempo_sin_pri_notific: tempoSinPriNotific,
        tempo_invest_encerrar: tempoInvestEncerrar,
        unidade: unidade
      })
    });

    const data = await response.json();
    if (response.ok) {
      document.getElementById('resultBox').classList.remove('oculto');
      document.getElementById('resultBox').innerText = `Previsão de dias até encerramento: ${data.previsao.toFixed(1)}`;
    } else {
      alert("Erro: " + data.detail);
    }
  } catch (err) {
    console.error(err);
    alert("Erro ao comunicar com o backend.");
  }
});

// Botão limpar
document.getElementById('resetBtn').addEventListener('click', () => {
  document.getElementById('dataSintoma').value = '';
  document.getElementById('dataNotificacao').value = '';
  document.getElementById('dataInvestigacao').value = '';
  document.getElementById('unidadeSelect').value = '';
  document.getElementById('resultBox').classList.add('oculto');
});
