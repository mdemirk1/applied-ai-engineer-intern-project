import { Component } from '@angular/core';
import { ChatService } from './services/chat.service';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = '⚡ ThunderBot';
  messages: Message[] = [];
  userInput = '';
  loading = false;
  darkMode = true;

  constructor(private chatService: ChatService) {}

  sendMessage(): void {
    const input = this.userInput.trim();
    if (!input) return;

    this.messages.push({ sender: 'user', text: input });
    this.userInput = '';
    this.loading = true;

    this.chatService.sendMessage(input).subscribe({
      next: (res: any) => {
        const reply = res?.answer ?? 'No answer.';
        this.loading = false;
        this.simulateTyping(reply);
      },
      error: () => {
        this.messages.push({ sender: 'bot', text: 'Error contacting server.' });
        this.loading = false;
      }
    });
  }

  simulateTyping(fullText: string) {
    const botMsg: Message = { sender: 'bot', text: '' };
    this.messages.push(botMsg);

    let i = 0;
    const interval = setInterval(() => {
      botMsg.text += fullText.charAt(i);
      i++;
      if (i >= fullText.length) clearInterval(interval);
    }, 20);
  }

  toggleTheme(): void {
    this.darkMode = !this.darkMode;
  }
}
