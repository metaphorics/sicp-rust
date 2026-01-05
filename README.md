# SICP-Rust

<img src="https://sicpebook.files.wordpress.com/2013/09/smile0.png"
 alt="Par smiling" align="right" />

이 프로젝트는 Abelson, Sussman, Sussman의 저서 "컴퓨터 프로그램의 구조와 해석 (Structure and Interpretation of Computer Programs, SICP)"의 새로운 HTML5 및 EPUB3 버전입니다. 이 버전은 MIT Press의 원본 [HTML 버전](https://mitpress.mit.edu/sicp)에서 변환된 [비공식 Texinfo 포맷 (Unofficial Texinfo Format)](http://www.neilvandyke.org/sicp-texi)의 계보를 잇고 있습니다.

<b>EPUB3 형식: [sicp.epub](https://github.com/sarabander/sicp-epub/blob/master/sicp.epub?raw=true)</b>

<b>온라인 읽기: [HTML 도서](https://sarabander.github.io/sicp)</b>

확장 가능한 벡터 그래픽(SVG), MathML 및 MathJax를 이용한 수학적 마크업, 내장 웹 폰트, 그리고 구문 강조(syntax highlighting)와 같은 현대적인 솔루션들이 사용되었습니다. 모바일 기기와 태블릿에서의 가독성을 높이기 위해 반응형 디자인이 적용되어 있습니다. 작은 화면에서의 폰트 크기 및 서식 조정을 위해 더 많은 테스트가 필요하므로, 스마트폰이나 태블릿 사용자분들의 피드백을 환영합니다.

## 소스 (Source)

루트 디렉토리에는 Texinfo 소스 파일인 `sicp-pocket.texi`가 들어 있습니다. HTML 파일을 재생성하고 EPUB을 빌드하려면 다음 명령을 입력하세요:

```bash
$ make
```

`html` 디렉토리 내의 파일들(하위 디렉토리 제외)은 모두 덮어씌워지므로, 수정을 원하신다면 `sicp-pocket.texi` 파일을 수정하는 것이 좋습니다. EPUB 파일은 프로젝트 트리 외부의 상위 디렉토리에 생성됩니다.

책을 컴파일하려면 [Texinfo 5.1](https://ftp.gnu.org/gnu/texinfo), Perl 5.12 이상, Ruby 1.9.3 이상, [Nokogiri](http://nokogiri.org) 젬, [PhantomJS](http://phantomjs.org), 그리고 인터넷 연결이 필요합니다.

## 감사의 말 (Acknowledgements)

- Lytha Ayth
- Neil Van Dyke
- Gavrie Philipson
- Li Xuanji
- J. E. Johnson
- Matt Iversen
- Eugene Sharygin

## 라이선스 (License)

소스 파일인 `sicp-pocket.texi`, 도서의 HTML 콘텐츠, 그리고 `html/fig` 디렉토리의 다이어그램들은 크리에이티브 커먼즈 저작자표시-동일조건변경허락 4.0 국제 라이선스([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)) 하에 배포됩니다.

대부분의 스크립트들은 GNU 일반 공중 사용 허가서(GPL) 버전 3 하에 배포됩니다 (자세한 내용은 LICENSE.src 참조).

폰트들은 SIL 오픈 폰트 라이선스(OFL) 버전 1.1을 따릅니다. 자바스크립트 라이브러리와 같은 다른 파일들은 각각의 라이선스를 따릅니다.

## 자매 프로젝트 (Sister project)

LaTeX 소스로부터 빌드된 [PDF 버전](https://github.com/sarabander/sicp-pdf)이 이 HTML 버전과 함께 제공됩니다.
