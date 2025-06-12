**Zadanie 2** Scala

:white_check_mark: 3.0 Należy stworzyć kontroler do Produktów

:white_check_mark: 3.5 Do kontrolera należy stworzyć endpointy zgodnie z CRUD - dane pobierane z listy

:white_check_mark: 4.0 Należy stworzyć kontrolery do Kategorii oraz Koszyka + endpointy zgodnie z CRUD

:white_check_mark: 4.5 Należy aplikację uruchomić na dockerze (stworzyć obraz) oraz dodać skrypt uruchamiający aplikację via ngrok

:white_check_mark: 5.0 Należy dodać konfigurację CORS dla dwóch hostów dla metod CRUD

[commit](https://github.com/wojciechp6/ebiznes/commit/a22d5c81ec1c585bd49a88ba11d368c481f18db9)

Folder: scala

**Zadanie 3** Kotlin

:white_check_mark: 3.0 Należy stworzyć aplikację kliencką w Kotlinie we frameworku Ktor,
która pozwala na przesyłanie wiadomości na platformę Discord

:white_check_mark: 3.5 Aplikacja jest w stanie odbierać wiadomości użytkowników z
platformy Discord skierowane do aplikacji (bota)

:white_check_mark: 4.0 Zwróci listę kategorii na określone żądanie użytkownika

:white_check_mark: 4.5 Zwróci listę produktów wg żądanej kategorii

:x: 5.0 Aplikacja obsłuży dodatkowo jedną z platform: Slack, Messenger,
Webex

[commit](https://github.com/wojciechp6/ebiznes/commit/b5244ad97f1049fb08478cb511de5e248ee88b0c)

Folder: kotlin

**Zadanie 4** Go

:white_check_mark: 3.0 Należy stworzyć aplikację we frameworki echo w j. Go, która będzie miała kontroler Produktów zgodny z CRUD

:white_check_mark: 3.5 Należy stworzyć model Produktów wykorzystując gorm oraz wykorzystać model do obsługi produktów (CRUD) w kontrolerze (zamiast listy)

:white_check_mark: 4.0 Należy dodać model Koszyka oraz dodać odpowiedni endpoint

:white_check_mark: 4.5 Należy stworzyć model kategorii i dodać relację między kategorią, a produktem

:white_check_mark: 5.0 pogrupować zapytania w gorm’owe scope'y

[commit](https://github.com/wojciechp6/ebiznes/commit/6b48cbd69ea756c5a2f874e1cf59f15589f128e8)

Folder: go

**Zadanie 5** Frontend

Należy stworzyć aplikację kliencką wykorzystując bibliotekę React.js.
W ramach projektu należy stworzyć trzy komponenty: Produkty, Koszyk
oraz Płatności. Koszyk oraz Płatności powinny wysyłać do aplikacji
serwerowej dane, a w Produktach powinniśmy pobierać dane o produktach
z aplikacji serwerowej. Aplikacja serwera w jednym z trzech języków:
Kotlin, Scala, Go. Dane pomiędzy wszystkimi komponentami powinny być
przesyłane za pomocą React hooks.

:white_check_mark: 3.0 W ramach projektu należy stworzyć dwa komponenty: Produkty oraz
Płatności; Płatności powinny wysyłać do aplikacji serwerowej dane, a w
Produktach powinniśmy pobierać dane o produktach z aplikacji
serwerowej; 

:white_check_mark: 3.5 Należy dodać Koszyk wraz z widokiem; należy wykorzystać routing 

:white_check_mark: 4.0 Dane pomiędzy wszystkimi komponentami powinny być przesyłane za
pomocą React hooks 

:white_check_mark: 4.5 Należy dodać skrypt uruchamiający aplikację serwerową oraz
kliencką na dockerze via docker-compose 

:white_check_mark: 5.0 Należy wykorzystać axios’a oraz dodać nagłówki pod CORS 

[commit](https://github.com/wojciechp6/ebiznes/commit/5e585b1478bab46d5f856d3a4c5f28733d8baf52)

Folder: frontend

**Zadanie 6** Testy

:white_check_mark: 3.0 Należy stworzyć 20 przypadków testowych w CypressJS lub Selenium
(Kotlin, Python, Java, JS, Go, Scala)

:white_check_mark: 3.5 Należy rozszerzyć testy funkcjonalne, aby zawierały minimum 50
asercji

:white_check_mark: 4.0 Należy stworzyć testy jednostkowe do wybranego wcześniejszego
projektu z minimum 50 asercjami

:white_check_mark: 4.5 Należy dodać testy API, należy pokryć wszystkie endpointy z
minimum jednym scenariuszem negatywnym per endpoint

:x: 5.0 Należy uruchomić testy funkcjonalne na Browserstacku



**Zadanie 7** Sonar

Należy dodać projekt aplikacji klienckiej oraz serwerowej (jeden
branch, dwa repozytoria) do Sonara w wersji chmurowej
(https://sonarcloud.io/). Należy poprawić aplikacje uzyskując 0 bugów,
0 zapaszków, 0 podatności, 0 błędów bezpieczeństwa. Dodatkowo należy
dodać widżety sonarowe do README w repozytorium dane projektu z
wynikami.

:white_check_mark: 3.0 Należy dodać litera do odpowiedniego kodu aplikacji serwerowej w
hookach gita [Link do commita 1](https://github.com/wojciechp6/ebiznes/commit/5ee9c3c2861f4cd8ed435ea037580effb5c82583)

:white_check_mark: 3.5 Należy wyeliminować wszystkie bugi w kodzie w Sonarze (kod
aplikacji serwerowej) [Link do commita 2](https://github.com/wojciechp6/ebiznes/commit/5ee9c3c2861f4cd8ed435ea037580effb5c82583)

:white_check_mark: 4.0 Należy wyeliminować wszystkie zapaszki w kodzie w Sonarze (kod
aplikacji serwerowej) [Link do commita 3](https://github.com/wojciechp6/ebiznes/commit/5ee9c3c2861f4cd8ed435ea037580effb5c82583)

:white_check_mark: 4.5 Należy wyeliminować wszystkie podatności oraz błędy bezpieczeństwa
w kodzie w Sonarze (kod aplikacji serwerowej) [Link do commita 4](https://github.com/wojciechp6/ebiznes/commit/5ee9c3c2861f4cd8ed435ea037580effb5c82583)

:x: 5.0 Należy wyeliminować wszystkie błędy oraz zapaszki w kodzie
aplikacji klienckiej [Link do commita 5]()


Kod: 

[Backend](https://github.com/wojciechp6/ebiznes-sonarcube-backend)

[Frontend](https://github.com/wojciechp6/ebiznes-sonarcube-frontend) 


**Zadanie 8** OAuth2

:white_check_mark: 3.0 logowanie przez aplikację serwerową (bez Oauth2)

:white_check_mark: 3.5 rejestracja przez aplikację serwerową (bez Oauth2)

:white_check_mark: 4.0 logowanie via Google OAuth2

:white_check_mark: 4.5 logowanie via Facebook lub Github OAuth2

:white_check_mark: 5.0 zapisywanie danych logowania OAuth2 po stronie serwera

[commit](https://github.com/wojciechp6/ebiznes/commit/8ae0b960cfb82fc4dd3cc6922dadec80e1fa5720)

Folder: oauth2


**Zadanie 9** ChatGPT bot

:white_check_mark: 3.0 należy stworzyć po stronie serwerowej osobny serwis do łącznia z chatGPT do usługi chat

:white_check_mark: 3.5 należy stworzyć interfejs frontowy dla użytkownika, który komunikuje się z serwisem; odpowiedzi powinny być wysyałen do frontendowego interfejsu

:white_check_mark: 4.0 stworzyć listę 5 różnych otwarć oraz zamknięć rozmowy

:white_check_mark: 4.5 filtrowanie po zagadnieniach związanych ze sklepem (np. ograniczenie się jedynie do ubrań oraz samego sklepu) do GPT

:white_check_mark: 5.0 filtrowanie odpowiedzi po sentymencie

[commit](https://github.com/wojciechp6/ebiznes/commit/e38f1cbdad4689b09e088fd531169543f50be35e)

Folder: chat

**Zadanie 10** Chmura/CI

:white_check_mark: 3.0 Należy stworzyć odpowiednie instancje po stronie chmury na dockerze

:white_check_mark: 3.5 Stworzyć odpowiedni pipeline w Github Actions do budowania aplikacji (np. via fatjar)

:white_check_mark: 4.0 Dodać notyfikację mailową o zbudowaniu aplikacji

:white_check_mark: 4.5 Dodać krok z deploymentem aplikacji serwerowej oraz klienckiej na chmurę

:x: 5.0 Dodać uruchomienie regresyjnych testów automatycznych
(funkcjonalnych) jako krok w Actions

[commit](https://github.com/wojciechp6/ebiznes/commit/665a92b7b2a72cb14fc7d63ff6b4ef90bae9b3b2)

Folder: cloud

[FRONTEND](https://ebiznes-frontend-wp6-c6f2fvdghcexccaz.polandcentral-01.azurewebsites.net)

[BACKEND](https://ebiznes-backend-wp6-agfwbuffe7cuhmg6.polandcentral-01.azurewebsites.net/api/products)
