# video_game_ind_project
---
## Project Description
Video games have been around for decades now, and they only get more and more popular as time goes on. The Consoles, characters, and the price may change but one thing has stayed throughout, and it's the stories they tell. Similar to tv and movies, the genre they fall into define the content and usually fans of a genre will flock to new content purely if it is in the same style. As an avid gamer and someone who's seen the power of analysis, I wanted to see if we could determine how well a game might do purely off the genre it belongs to. This project will perform surface level analysis on the insights of genre in 3 different regions of the world as well as a global overview with the hope of conducting more in depth analysis as time progresses.
---
## Project Goals
- Determine best selling genres
- Apply regionality to observe potential change in sales
---
## Initial Hypotheses
- I've always had the assumption that the genre 'Shooter' was among the most popular here. Shooter being the colloquial name for a game where your "character" or "avatar" is, well, shooting at something. Whether it's an older game like starfox where you pilot a spaceship fighting opposing ships, or a newer game like Destiny where you control a character's running and jumping from planet to planet fighting "the forces of evil", just about everyone has either played or knows someone who has.
- Looking to outside regions, my initial assumption was that more modern shooters like the "call of duty" franchise would see a huge falloff outside of North America and we'd see a rise in other genres like racing in Europe and platformers in Japan
---
## Data Dictionary
| Feature | Definition | 
| :- | :- |
| Platform | Console game released on, 0 = Retro, 1 = Home Console, 2 = Handheld |
| year_of_release | The year the game was released |
| <font color='red'>Genre</font> | The category of Genre the game falls into, 0 = 'Action', 1 = 'Sports', 2 = 'Misc', 3 = 'RPG', 4 = 'Shooter', 5 = 'Adventure', 6 = 'Racing', 7 = 'Platform', 8 = 'Simulation', 9 = 'Fighting', 10 = 'Strategy', 11 = 'Puzzle' |
| na_sales | The amount of games sold in North American Region |
| eu_sales | The amount of games sold in European Region |
| jp_sales | The amount of games sold in Japanese Region |
| global_sales | The amount of games sold globally |
---
## Steps to Reproduce
- included in this repository is the initial csv before cleaning as well as the cleaned csv to get started
- a support.py file will contain all the functions used in the final_report
