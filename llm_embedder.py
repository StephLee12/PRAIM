import numpy as np
import torch
import logging

from typing import Dict, List, Any

from angle_emb import AnglE, Prompts
from utils import EVChargingDataPoint




class LLMEmbedder:
    """Class to generate embeddings from EV charging data using LLM."""
    
    def __init__(
        self, 
        device: torch.device,
        model_name: str = "models/llama-2-7b-hf", 
        lora_path: str = "models/angle-llama-7b-nli-v2"
    ):
        """
        Initialize the LLM embedder.
        
        Args:
            model_name: Name or path of the LLM model to use with AnglE
            lora_path: Path to the LoRA adapter for AnglE
            prompt_template: Custom prompt template (if None, uses default)
            use_angle: Whether to use AnglE embedding framework
            llm_model: Pre-trained LLM model (used if use_angle=False)
            tokenizer: Tokenizer for the LLM model (used if use_angle=False)
        """
        
        self.logger = logging.getLogger("LLMEmbedder")
        self.logger.info(f"Loading AnglE model from {model_name} with LoRA {lora_path}")
        
        # Initialize AnglE model
        self.angle = AnglE.from_pretrained(model_name, pretrained_lora_path=lora_path)
        # self.angle.set_prompt(prompt=Prompts.A)
        
        # Get device
        self.device = device
        
        # Default template resembles your EV charging context
        self.prompt_template = """
        You are an AI assistant specializing in data analysis.

        Please analyze the following electric vehicle (EV) charging station data, with a focus on identifying and imputing missing values:

        [EV_CHARGING_DATA]

        Consider, but do not limit yourself to, the following aspects:  
        1. **Temporal patterns** (daily, weekly, seasonal variations)  
        2. **Station-specific characteristics**  
        3. **Correlations between calendar features and charging demand**  
        4. **Similar historical charging patterns**  
        5. **Contextual factors influencing usage trends**  

        Based on your analysis, propose reasonable estimates for missing values and assess the confidence level of these estimates.
        """

        
            

    
    def format_prompt(
        self, 
        history: np.ndarray, 
        missing_mask: np.ndarray,
        station_info: Dict[str, Any],
        calendar_info: Dict[str, Any]
    ) -> str:
        """
        Format EV charging data for prompt.
        
        Args:
            history: Historical charging demand data
            missing_mask: Boolean mask indicating missing values
            station_info: Dictionary of station metadata
            calendar_info: Dictionary of calendar information
            
        Returns:
            Formatted EV data string
        """
        # Format historical data with missing indicators
        history_str = "Historical charging demand (kWh):\n"
        for i, (val, is_missing) in enumerate(zip(history, missing_mask)):
            day_offset = i - len(history) + 1
            if day_offset == 0:
                day_str = "Today"
            elif day_offset == -1:
                day_str = "Yesterday"
            else:
                day_str = f"{abs(day_offset)} days ago"
                
            if is_missing:
                history_str += f"{day_str}: MISSING\n"
            else:
                history_str += f"{day_str}: {val[0]:.2f}\n"
        
        # Format station information
        station_str = "Station information:\n"
        station_str += f"ID: {station_info.get('id', 'Unknown')}\n"
        station_str += f"Location: {station_info.get('location', 'Unknown')}\n"
        station_str += f"Type: {station_info.get('type', 'Unknown')}\n"
        if 'coordinates' in station_info:
            station_str += f"Coordinates: {station_info['coordinates']}\n"
        if 'surrounding_pois' in station_info:
            station_str += f"Nearby: {', '.join(station_info['surrounding_pois'])}\n"
        
        # Format calendar information
        calendar_str = "Calendar information:\n"
        calendar_str += f"Date: {calendar_info.get('year', 'Unknown')}-{calendar_info.get('month', 'Unknown')}-{calendar_info.get('day', 'Unknown')}\n"
        calendar_str += f"Day of week: {calendar_info.get('day_of_week', 'Unknown')}\n"
        is_weekend = calendar_info.get('is_weekend', False)
        calendar_str += f"Weekend: {'Yes' if is_weekend else 'No'}\n"
        if 'is_holiday' in calendar_info:
            calendar_str += f"Holiday: {'Yes' if calendar_info['is_holiday'] else 'No'}\n"
        
        # Combine all parts
        return f"{history_str}\n{station_str}\n{calendar_str}"
    
    
    
    def generate_embedding(self, data_point: 'EVChargingDataPoint') -> np.ndarray:
        """
        Generate embedding for a data point.
        
        Args:
            data_point: EVChargingDataPoint to generate embedding for
            
        Returns:
            Embedding array
        """
        # Format prompt by creating EV data string
        ev_data_str = self.format_prompt(
            data_point.history,
            data_point.missing_mask,
            data_point.station_info,
            data_point.calendar_info
        )
        
        # Replace the placeholder in the template with actual EV data
        # This is the key step similar to llm_model.py
        full_prompt = self.prompt_template.replace("[EV_CHARGING_DATA]", ev_data_str)
        
        # Generate embedding using AnglE
        with torch.no_grad():
            embedding = self.angle.encode({'text': full_prompt}, to_numpy=True, prompt=Prompts.A)[0]
        
        return embedding
    
    
    
    def batch_generate_embeddings(
        self, 
        data_points: List['EVChargingDataPoint'],
        batch_size: int=8
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple data points in batches.
        
        Args:
            data_points: List of EVChargingDataPoint objects
            batch_size: Number of points to process at once
            
        Returns:
            List of embedding arrays
        """
        embeddings = []
        
        for i in range(0, len(data_points), batch_size):
            batch = data_points[i:i+batch_size]
            
            # Process one by one if not using AnglE
            for dp in batch:
                embeddings.append(self.generate_embedding(dp))
            
            # if self.use_angle:
            #     # Format all prompts
            #     prompts = [self.format_prompt(
            #         dp.history, dp.missing_mask, dp.station_info, dp.calendar_info
            #     ) for dp in batch]
                
            #     # Generate embeddings in batch
            #     with torch.no_grad():
            #         batch_embeddings = self.angle.encode(
            #             [{'text': prompt} for prompt in prompts], 
            #             to_numpy=True
            #         )
                
            #     embeddings.extend(batch_embeddings)
            # else:
                # # Process one by one if not using AnglE
                # for dp in batch:
                #     embeddings.append(self.generate_embedding(dp))
        
        return embeddings

