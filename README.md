Эффект Pop Art 
==============

Реализация эффекта Pop Art с сайта befunky.com. 

Упрощенный алгоритм работы
---------------
Создание фона:
* Исходное изображение размывается.
* Из исходного изображения вычитаем размытое.
* Повторяем первые два пункта.
* Переводим полученное изображение в черно-белое (назовем его sharpen_bg, далее оно нам понадобится)
* Применяем функцию colormap (предварительно создаем соответствующую таблицу, чтобы было идентично оригинальному эффекту)  

Создание эффекта halftone поверх фона:
* Исходное изображение переводится в черное-белое.
* Создается маска, состоящая из кругов различных радиусов. Радиус зависит от цвета пикселя на черно-белом изображении. Чем темнее пиксель, тем больше радиус. Круги располагаются по сетке, шаг которой зависит от размера изображения.
* С помощью полученной маски копируем sharpen_bg и применяем функцию colormap (уже с другой таблицей).
* Полученное изображение копируем с помощью маски на фоновое.




Использование
-------------
Программа запускается в консольном режиме. Необходимо указать путь к изображению.
```bash
$ befunky -path

```

Ньюансы
-------
Что было подмечено при анализе данного эффекта:
* Радиус размытия при создании фона зависит от размера изображения. 
* Если изображение превышает по одной размерности 4088 пикселей, то оно будет уменьшено, чтобы самая длинная сторона не превышала указанную величину.
* Если изображение меньше 256 пикселей хотя бы по одной размерности, то оно уже не размывается, на нем не появляется halftone эффект. Также изменяется схема отображения цветов. Скорее всего, его просто переводят в черно-белое, умножают на константу, зависящую от его размера, и применяют ту же самую функцию colormap. Данная часть не была реализована.
* Также есть небольшое различие в эффекте halftone в моей реализации и оригинале. В оригинале края кругов немного размыты и имеют другой цвет. Но это не сильно заметно, поэтому было решено пренебречь данной разницей.