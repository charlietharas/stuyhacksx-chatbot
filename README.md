# stuyhacksx-chatbot
Project for StuyHacks X by Charlie Tharas and Jason Wu, Hunter College High School class of 2023. Easy open-source personalized Discord chatbot.

This is an open-source platform for creating a Discord bot that can communicate with users based on customized message history--essentially, this bot allows you to mimic people based on their own chat history, which can be a fun toy for any personal server, or even a more practical application in some use cases. At the most basic level, it is incredibly simple to turn exported chat data into a fully functional Discord application, but it also allows full customized tuning of model parameters. 

## Wiki
Our wiki offers a relatively detailed description of the few steps needed to get from simply owning a Discord account to operating a Discord chatbot. Below is a basic description of the simplest syntax required to train and execute the bot, provided you have the data.

## Installing and Dependencies
Preparing the data necessary to train the bot requires [this open source Discord exporter.](https://github.com/Tyrrrz/DiscordChatExporter) Download the latest release from their GitHub repository and follow the instructions built into the program/on their respective wiki to export your data into a .csv file. Any data preprocessing, filtering, or other tinkering you wish to do is entirely up to you!

This project also requires installing several packages via pip. We recommend using a virtual environment manager such as Anaconda for this process. Install the required dependencies with `pip install -r requirements.txt`.

Note that the project does not support the usage of a GPU to train at this time.

## Training
If you have the data .csv file prepared already,  the following command will train the model with default parameters. The only things required are the exported data, user ID of the user you wish to mimic, and a save directory.

    python train.py -f filename.csv -u user#1234 -save_dir save/filepath
To view the full list of flags, run `python train.py --help` or see our wiki!

This model also supports the use of [TensorDash](https://github.com/CleanPegasus/TensorDash) via the addition of your TensorDash password and email with the `--tensordash_email EMAIL` and `--tensordash_password PASSWORD`flags. The project's default name is 'Chatbot'. 
For debugging or analyzing additional features of the data, consider adding the `--feature_analysis` flag, and further customizing that with `--words WORDS`. 

Note that exported files may be relatively large, often ranging around 150-400MB in size *per checkpoint.*

## Executing the Bot
If you've never created a Discord bot or application before, a tutorial is available [here.](https://www.freecodecamp.org/news/create-a-discord-bot-with-python/) You can disregard the tutorial after the section that reads "How to Code a Basic Discord Bot with the discord.py library" as the bot's code is included in the source code here. You **will** require the token generated from your bot to run the bot's code here.
To run the bot, you will need your original .csv file, the model's exported .tar file, the user ID of the user you wish to mimic, and your Discord token.

    python run.py -f filename.csv -lf checkpoint.tar -u user#1234 -t TOKEN
This program will need to be perpetually running as long as you wish for your Discord bot to be online.

## Project Notes
This project was coded over Google Colaboratory and PyDev for Eclipse by Charlie Tharas and Jason Wu for the StuyHacks X hackathon hosted by Stuyvesant High School (2021). The [NLP model](https://medium.com/swlh/end-to-end-chatbot-using-sequence-to-sequence-architecture-e24d137f9c78) used in this project was adapted to fit .csv data and pipelined into code for a Discord bot. Please note that Discord Chat Exporter is in apparent violation of Discord's Terms of Service and use caution when engaging in such practices. **Never** share your personalized user token or your bot's token with anyone you do not trust.
