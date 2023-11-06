# 📷 image-detection

Este script `main.py` foi desenvolvido para processar dados de vídeo e realizar a detecção de objetos usando o modelo YOLOv8. Ele extrai objetos de um vídeo, desenha caixas delimitadoras ao redor deles e pode opcionalmente salvar o vídeo processado. Abaixo, você encontrará uma breve descrição do script e como usá-lo.

## 🛠️ Pré-requisitos

Antes de usar o script, você deve ter os seguintes pré-requisitos instalados:

1. Python 3.x
2. OpenCV (`cv2`)
3. Modelo YOLOv8 (é baixado automaticamente ao executar o script)
4. Ultralytics

## ▶️ Uso

Você pode usar o script seguindo estas etapas:

1. Clone o repositório ou baixe o script (`main.py`) para o seu ambiente local.

2. Certifique-se de ter um arquivo de vídeo chamado `video.mp4` no mesmo diretório que o script. Se o arquivo de vídeo estiver localizado em outro lugar, atualize a variável `source` com o caminho do arquivo correto.

3. Execute o script com o seguinte comando:

      ```bash
   python main.py

  Opcionalmente, você pode passar alguns argumentos da linha de comando para modificar o comportamento do script:
  
  `--view_img`: Esse sinalizador, quando fornecido, exibirá o vídeo processado com caixas delimitadoras em tempo real.
  
  `--save_img`: Esse sinalizador, quando fornecido, salvará o vídeo processado com caixas delimitadoras no diretório output.
  
  Exemplo de comando com sinalizadores adicionais:
  

    python main.py --view_img --save_img

- Você pode personalizar o limiar de confiança e outros parâmetros, modificando a função `main` no script para atender aos seus requisitos específicos.

- Sinta-se à vontade para modificar o script e seus parâmetros para adaptá-lo ao seu próprio caso de uso.

Certifique-se de instalar a biblioteca `ultalytics` antes de executar o script:


    pip install ultalytics


- Você pode personalizar o limiar de confiança e outros parâmetros, modificando a função `main` no script para atender aos seus requisitos específicos.

- Sinta-se à vontade para modificar o script e seus parâmetros para adaptá-lo ao seu próprio caso de uso.


