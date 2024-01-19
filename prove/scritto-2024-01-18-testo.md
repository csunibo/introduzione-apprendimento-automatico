# Selezionare la sentenza corretta riguardo agli alberi di decisione
1. Possono espimere qualunque funzione di classificazione
2. Non presentano problemi di overfitting
3. Possono essere utilizzati solo con features discrete
4. Il costo computazionale della predizione è piuttosto elevato

# Selezionare la sentenza ERRATA relativa all'entropia per la distribuzione di probabilità di una variabile aleatoria discreta
1. è una misura del grado di disordine della variabile aleatoria
2. Il range del suo valore è tra 0 e log n dove n sono i possibili valori di X
3. Il suo valore è minimo (e uguale a 0) quando la probabilità è tutta concentrata in una classe
4. Il suo valore è minimo (e uguale a 0) quando la probabilità è equamente distribuita tra tutte le classi

# Se un modello calcola una distribuzione di probabilità, aggiungere alla funzione obiettivo una componente tesa a diminuire l'entropia avrà l'effetto di:
1. favorire l'uscita da minimi locali
2. ridistribuire le probabilità in modo più bilanciato tra tutti i casi
3. nessun effetto concreto
4. focalizzare le scelte sui casi più probabili

# Selezionare la sentenza ERRATA relativa alla derivata della funzione logistica:
1. tende a 0 quando x tende a meno infinito
2. è una funzione monotona
3. ha il suo massimo in corrispondenza dello 0
4. è una funzione simmetrica 

# Ci sono due dadi, uno normale e uno truccato che restituisce un 6 con probabilità .5, e gli altri valori con probabilità 0.1. Faccio tre lanci con lo stesso dado e osservo un 3, un 6 e un 2. Cosa posso concludere?
1. E' più probabile che il dato sia normale
2. Nulla
3. La probabilità di usare uno o l'altro dei dadi è esattamente la stessa
4. E' più probabile che il dato sia truccato

# Selezionare la sentenza ERRATA riguardo all'apprendimento supervisionato
1. Si riferisce all'apprendimento di funzioni basato su esempi di training composti da coppie input-output
2. Può comprendere sia problemi di regressione che di classificazione
3. Richiede la costante supervisione di un esperto durante il training
4. La definizione della ground truth può richiedere l'intervento umano ed essere onerosa

# Selezionare la sentenza ERRATA relativa alla tecnica Naive Bayes
1. Deriva dall'ipotesi teorica semplificativa che le features siano indipendenti tra loro, date le classi
2. Fornisce un modo computazionalmente efficiente per approssimare la distribuzione congiunta di probabilità delle features
3. E' una tecnica di tipo generativo in quanto cerca di determinare la distribuzione delle varie categorie dei dati
4. Non può essere utilizzata se le features non sono tra loro indipendenti, date le classi

# Un dataset contiene 1/3 di positivi e 2/3 di negativi. La recall del modello è di 2/3. Che percentuale dei dati sono Falsi Positivi?
1. 1/3
2. 1/9
3. Non può essere stabilito
4. 2/9

# Selezionare la sentenza SCORRETTA riguardo alla regressione logistica
1. Permette di associare una probabilità alla predizione della classe
2. Il calcolo della predizione si basa sulla loglikelihood dei dati di training
3. I parametri del modello possono essere tipicamente calcolati in forma chiusa, mediante una formula esplicita
4. La predizione dipende dal bilanciamento dei dati di training rispetto alle classi

# Selezionare la sentenza SCORRETTA relativa alla tecnica a discesa del gradiente.
1. Il risultato può dipendere dalla inizializzazione dei parametri del modello
2. E' opportuno decrementare il learning rate verso la fine dell'apprendimento
3. Potrebbe convergere a un minimo locale
4. Può essere applicata solo se la funzione da minimizzare ha una superficie concava

# Quale è l'effetto tipico dell'aumento della dimensione del minibatch durante il training?
1. La Backpropagation è effettuata meno frequentemente ma l'aggiornamento dei parametri è più accurato
2. La Backpropagation è effettuata più frequentemente e l'aggiornamento dei parametri è più accurato
3. La Backpropagation è effettuata più frequentemente ma l'aggiornamento dei parametri è meno accurato
4. La Backpropagation è effettuata meno frequentemente e l'aggiornamento dei parametri è meno accurato

# Riguardo alla regressione multinomiale, selezionare la sentenza corretta tra le seguenti
1. Il peso con cui è valutata ogni feature è tipicamente diverso per ogni classe
2. Per n features di input e m classi, il numero dei parametri del modello cresce come O(n+m)
3. Per ogni input, esiste almeno una classe con probabilità > 0.5
4. I pesi delle features sono sempre tutti positivi, i bias possono essere negativi

# Selezionare la sentenza corretta relativa alla funzione softmax
1. Restituisce una distribuzione di probabilità sulle classi
2. Produce valori compresi nell'intervallo [-1,1]
3. Per una data classe,  la somma dei valori su tutti gli input di un minibatch è sempre 1
4. Non può essere utilizzata nel caso di una classificazione binaria

# In quale di questi casi una tecnica di classificazione lineare potrebbe non fornire risultati soddisfacenti:
1. Quando esiste una elevata correlazione tra le features 
2. Quando non tutte le features di input sono rilevanti ai fini della classificazione
3. Quando la classificazione dipende da un confronto tra features 
4. Quando le features sono indipendenti tra loro, data la classe.

# Selezionare la sentenza corretta relativa alle tecniche discriminative
1. Si focalizzano sulla definizione delle frontiere di decisione (decision boundaries)
2. Cercano di determinare le distribuzioni di probabilità delle varie classi di dati
3. Sono tipicamente meno espressive delle tecniche generative
4. Si applicano per lo più in ambito di apprendimento non supervisionato

# Selezionare la sentenza SCORRETTA relativa all'overfitting
1. Può essere particolarmente pericolosa per modelli altamenti espressivi
2. L'acquisizione di nuovi dati di training non può che peggiorare la situazione
3. Può essere contrastata con tecniche di regolarizzazione
4. Può essere contrastata con la tecnica di early stopping durante la fase di training

# Selezionare la sentenza ERRATA relativa alla funzione ReLU(x) (rectified linear unit):
1. non può essere utilizzata per layer convoluzionali
2. La sua derivata è una funzione a gradino
3. Lei o le sue varianti sono tipicamente utilizzate per i livelli interni delle reti neurali profonde
4. è una funzione monotona non decrescente 

# Selezionare la sentenza SCORRETTA relativa ai neuroni artificiali
1. Il numero dei parametri di un neurone artificiale è lineare nel numero dei suoi input
2. Un neurone artificiale può apprendere qualunque funzione dei suoi input
3. Un neurone artificiale tipicamente calcola una combinazione lineare dei suoi input, seguita dalla applicazione di una funzione di attivazione non lineare
4. Un neurone artificale definisce un semplice modello matematico che simula il neurone biologico

# Selezionare la sentenza SCORRETTA relativa alla backpropagation per reti neurali
1. Ha un costo computazionale paragonabile a quello del calcolo "in avanti" (inference) lungo la rete
2. Si basa tipicamente su algoritmi di tipo genetico
3. Tipicamente, si effettua solo durante la fase di "training" della rete
4. Richiede la memorizzazione delle attivazioni di tutti i neuroni della rete durante la forward pass

# Selezionare la sentenza SCORRETTA relativa al problema della scomparsa del gradiente (vanishing gradient) 
1. Il problema è fortemente attenuato dall'uso di ReLU (o sue varianti) come funzione di attivazione per i livelli nascosti della rete
2. Se il gradiente tende a zero anche i parametri e le attivazioni dei neuroni tendono a zero
3. Se il gradiente tende a zero i parametri non sono più aggiornati e la rete smette di apprendere
4. Il problema è mitigato dall'uso di link residuali all'interno della rete

# Quale funzione di loss è tipicamente utilizzata  in una rete neurale per classificazione binaria che utilizza una sigmoid come attivazione finale?
1. binary crossentropy
2. categorical crossentropy
3. absolute error
4. mean squared error

# Un layer convolutivo 2D con stride 1, kernel size 3x3, e senza padding prende in input un layer con dimensioni (32,32,3) e restituisce un layer di dimensione (32,32,16). Quanti sono i suoi parametri?
1. 448
2. 432
3. 28
4. 160

# Il tensore di input di un layer convolutivo 2D ha dimensione (16,16,32). Sintetizzo 8 kernel con dimensione spaziale (3,3), stride 2, nessun padding (valid mode). Quale sarà la dimensione dell'output?
1. (7,7,8)
2. (8,8,32)
3. (8,8,8)
4. (7,7,15)

# Qual'è l'effetto di uno stride non unitario (>1) in un layer convolutivo?
1. La dimensione spaziale aumenta
2. Nessun effetto spaziale, il numero dei canali aumenta
3. La dimensione spaziale diminuisce
4. Nessun effetto spaziale, il numero dei canali decresce

# Selezionare la sentenza scorretta relativa al campo ricettivo (receptive field) di un neurone di una CNN:
1. E' sempre almeno pari alla dimensione spaziale del dato di input
2. Dipende dalla profondità del layer in cui si trova il neurone e dalle dimensioni e gli strides dei kernel dei layers precedenti
3. Aumenta rapidamente con l'attraversamento di livelli con downsampling 
4. Definisce la porzione dell'input che influenza l'attivazione di un determinato neurone

# Selezione al sentenza ERRATA relativa alle Transposed Convolutions
1. Possono essere interpretate come convoluzioni normali con stride sub-unitario
2. Sono prevalentemente utilizzate in architetture per Image-to-Image processing, come autoencoders o U-Nets
3. Richiedono la trasposizione dell'input prima di calcolare la convoluzione del Kernel
4. Sono essenzialmente equivalenti alla applicazione di un livello di upsampling seguito da una convoluzione normale

# Quale delle seguenti sentenze relative agli autoencoders è SCORRETTA?
1. L'encoder e il decoder non devono essere necessariamente simmetrici
2. Possono essere utilizzate per la rimozione di rumore (denoising)
3. La rappresentazione interna prodotta dall'encoder abitualmente ha una dimensione ridotta rispetto a quella di partenza
4. Gli Autoencoders richiedono l'uso di livelli densi

# Selezionare la sentenza SCORRETTA riguardo ai modelli generativi
1. Sono modelli che cercano di apprendere la distribuzione di probabilità dei dati
2. Un tipico esempio di tecnica generativa è Naive Bayes 
3. Generative Adversarial Networks, Variational Autoencoders e Diffusion models sono esempi di tecniche generative profonde
4. Sono modelli meta-teorici rivolti alla automatizzazione della generazione di reti neurali 

# Quale di queste reti NON è stata progettata per la classificazione di immagini?
1. Inception-v3
2. ResNet
3. U-Net
4. VGG19

# Selezionare la sentenza SCORRETTA relativa alla U-Net
1. E' un componente tipico dei modelli generativi a diffusione
2. Può essere usata per la rimozione del rumore (denoising) di immagini
3. E' spesso impiegata per problemi di segmentazione semantica di immagini
4. Viene spesso utilizzata nell'ambito della classificazione dei generi musicali

