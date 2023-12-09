import os
import codecs
import pandas as pd
import numpy as np


def get_human_generated_labels(seed, num_to_run, demo=False):
    """Randomly selects a subset of faculty bios to retrieve for human evaluation. Prints faculty bio text to command line for
    user to manually evaluate. Prompts user to indicate if bio contains a name ("1" = yes, "0" = no), then prompts user to
    indicate if bio contains an email address ("1" = yes, "0" = no).

    Code author (Anna Otte) annotated bios when seed = 0 and num_to_run = 100, so if user sets seed 0 with 100 bios or less as
    the input for num_to_run, then function will fast track retrieving the human generated labels from a pre-existing csv file
    rather than prompting user for input.

    Args:
        seed (int): positive integer indicating random seed to run
        num_to_run (int): positive integer indicating number of faculty bios to retrieve for human evaluation
        demo (bool): indicates if demo is being run for peer-review purposes

    Returns:
        bios_to_run (list): faculty bio numbers that were evaluated
        has_name (list): 1s/0s (human-generated labels) indicating whether bio has/does not have a name. Order corresponds to bios_to_run.
        has_email (list): 1s/0s (human-generated labels) indicating whether bio has/does not have an email address. Order corresponds to bios_to_run.
    """
    if seed == 0 and num_to_run <= 100 and not demo:
        # code author already labeled first 100 choices when random seed is set to 0, so this conditional fast-tracks
        # the retrieval of the human generated labels for this specific set of inputs
        labels_df = pd.read_csv(
            os.path.join("test_results", "author_generated_labels.csv")
        )
        bios_to_run = labels_df["Bio_Num"].to_list()[:num_to_run]
        has_name = labels_df["Has_Name"].to_list()[:num_to_run]
        has_email = labels_df["Has_Email"].to_list()[:num_to_run]

    else:
        # set random seed and select subset of faculty bio numbers to evaluate
        np.random.seed(seed)
        bios_to_run = np.random.choice(6524, num_to_run)

        # set input filepath and initiate empty lists for name and email labels
        bios_path = os.path.join("data", "compiled_bios")
        has_name = []
        has_email = []

        for i in bios_to_run:
            # read faculty bio as string with UTF-8 encoding
            file_path = os.path.join(bios_path, str(i) + ".txt")
            with codecs.open(file_path, encoding="utf-8", errors="ignore") as f:
                bio = f.read()

            # print bio string for user to manually evaluate
            print(bio)

            # ask user if bio has a faculty name (1 = yes, 0 = no)
            has_name.append(
                int(
                    input(
                        "Does bio have a name? If yes, type 1 then press enter. Else, type 0 then press enter."
                    )
                )
            )

            # ask user if bio has a faculty email (1 = yes, 0 = no)
            has_email.append(
                int(
                    input(
                        "Does bio have a email? If yes, type 1 then press enter. Else, type 0 then press enter."
                    )
                )
            )

        # save human-generated labels with corresponding bio number to csv file for later viewing if desired
        human_generated_labels = pd.DataFrame(
            list(zip(bios_to_run, has_name, has_email)),
            columns=["Bio_Num", "Has_Name", "Has_Email"],
        )
        human_generated_labels.to_csv("test_results/user_generated_labels.csv")

    return bios_to_run, has_name, has_email
