from discord import SyncWebhook
import os


class EmptyWebhookLinkError(ValueError):
    pass


def send_discord_message(message, webhook_link=os.environ.get("WEBHOOK_LINK", "")):
    if webhook_link == "":
        raise EmptyWebhookLinkError("You should set-up WEBHOOK_LINK environment variable to be able to send messages "
                                    "on the discord webhook. For more information: "
                                    "https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks")

    webhook = SyncWebhook.from_url(webhook_link)
    webhook.send(message)


if __name__ == "__main__":
    send_discord_message("message de test.", "your_webhook_url_here")
