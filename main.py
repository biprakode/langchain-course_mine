import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


def main():
    information = """Sportacus 10 (Icelandic: Íþróttaálfurinn, lit. 'the Athletic Elf') is a fictional character from the Icelandic children's television show LazyTown, created and portrayed by Magnús Scheving. His name in English is a portmanteau of the ancient figure Spartacus and the word sport, which represents his athleticism. Sportacus humbly describes himself as a "slightly above-average hero", though his friends have a higher opinion of him, calling him a "superhero".

    Sportacus encourages the children of LazyTown to eat fruits and vegetables (which he calls "sports candy") and play outside instead of sitting around indoors and eating unhealthy food.[2] He wants to make sure LazyTown is happy, and knows that its residents have to be healthy and fit if they want to be happy. He is opposed by the sinister (yet equally energetic) Robbie Rotten, who seeks to return LazyTown to its former state: a lazy town. Sportacus is so engaged in his life of physical activity that he does parkour just to get from place to place—even doing acrobatic flips just to get from one side of his kitchen table to the other—and the children have to instruct him on how to relax.
    
    Sportacus lives in a large airship above LazyTown, which contains his bed, food, and other equipment, including a signed autograph from Jackie Chan. This is an Easter egg to the actor's portrayal of the villain in The Spy Next Door. """ # Sportacus wiki
    summary_template = """given the {information} about a person create the following
    1) A short summary
    2) 2 interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template,
    )

    llm = ChatGroq(temperature=0 , model="llama-3.1-8b-instant")

    chain = summary_prompt_template | llm
    reponse = chain.invoke(input = {"information": information})

    print(reponse.content)


if __name__ == "__main__":
    main()
